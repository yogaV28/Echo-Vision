import os
import logging
import tempfile
import subprocess
import ctypes
import pyaudio
import numpy as np
from gtts import gTTS
from faster_whisper import WhisperModel

# ----------------------------------------------------
# 1. Suppress ALSA Hardware Noise & Driver Logs
# ----------------------------------------------------
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)
def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EchoVoiceInterface:
    def __init__(self, language="en", model_size="tiny"):
        """
        Supports 'en' (English) and 'ta' (Tamil) for both STT (VRS) and TTS.
        """
        self.language = language

        # Initialize Faster-Whisper on Jetson GPU (CUDA) for instant VRS
        try:
            logging.info(f"[VRS Engine] Loading Whisper '{model_size}' model onto GPU (CUDA)...")
            self.stt_model = WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception as e:
            logging.warning(f"[VRS Engine] GPU initialization fallback to CPU: {e}")
            self.stt_model = WhisperModel(model_size, device="cpu", compute_type="int8")

        self.pa = pyaudio.PyAudio()
        self.input_device_index = self._get_optimal_mic_index()

    def _get_optimal_mic_index(self):
        """Finds system default / PulseAudio microphone stream."""
        for i in range(0, self.pa.get_host_api_info_by_index(0).get("deviceCount", 0)):
            dev_info = self.pa.get_device_info_by_host_api_device_index(0, i)
            if "pulse" in dev_info.get("name", "").lower():
                return i
        try:
            return self.pa.get_default_input_device_info().get("index")
        except Exception:
            return None

    def speak(self, text):
        """
        Google TTS (gTTS) pipeline with fallback to espeak-ng.
        Automatically switches voice language between Tamil (ta) and English (en).
        """
        if not text:
            return
        print(f"\n[John]: {text}")

        # Bilingual detection: check if text contains Tamil Unicode characters
        has_tamil_char = any("\u0b80" <= char <= "\u0bff" for char in text)
        tts_lang = "ta" if (has_tamil_char or self.language == "ta") else "en"

        temp_path = None
        try:
            # 1. Generate human-quality voice using Google TTS
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                temp_path = fp.name
                tts.save(temp_path)

            # 2. Play MP3 directly via PulseAudio system default sink
            subprocess.run(["mpg123", "-q", temp_path], check=True)

        except Exception as e:
            logging.warning(f"[TTS Fallback] Google TTS unavailable ({e}), using offline engine.")
            voice_code = "ta" if tts_lang == "ta" else "en-us"
            try:
                subprocess.run(["espeak-ng", "-v", voice_code, "-s", "160", "-a", "200", text], check=True)
            except Exception as espeak_err:
                logging.error(f"[TTS Error] Both gTTS and fallback failed: {espeak_err}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def listen_for_keyword(self, timeout_seconds=4, interrupt_callback=None, volume_threshold=400):
        """
        Captures audio with Voice Activity Detection (VAD) and transcribes in EN or TA.
        """
        chunk_size = 1024
        sample_rate = 16000

        try:
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input_device_index=self.input_device_index,
                input=True,
                frames_per_buffer=chunk_size,
            )
        except Exception as e:
            logging.error(f"[Mic Error] Failed to open audio stream: {e}")
            return ""

        frames = []
        has_spoken = False
        silence_chunks = 0
        max_loop_iterations = int((sample_rate / chunk_size) * timeout_seconds)

        for _ in range(max_loop_iterations):
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)

                # Compute RMS Volume for speech detection
                data_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(np.square(data_np)))

                if rms > volume_threshold:
                    if interrupt_callback and not has_spoken:
                        interrupt_callback()
                    has_spoken = True
                    silence_chunks = 0
                elif has_spoken:
                    silence_chunks += 1
                    # Stop after ~0.8s of silence once speech starts
                    if silence_chunks > 12:
                        break
            except Exception:
                break

        stream.stop_stream()
        stream.close()

        if not has_spoken or not frames:
            return ""

        audio_bytes = b"".join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            # Transcribe with language constraint (en/ta) or let whisper detect
            lang = self.language if self.language in ["en", "ta"] else None
            segments, _ = self.stt_model.transcribe(
                audio_np,
                beam_size=1,
                language=lang,
                vad_filter=True
            )
            transcription = " ".join([s.text for s in segments]).strip()
            return transcription
        except Exception as e:
            logging.error(f"[VRS Error] Transcription failed: {e}")
            return ""

    def parse_yes_no(self, phrase):
        """Bilingual confirmation parsing for Tamil and English."""
        if not phrase:
            return "NO"
        clean = phrase.lower().strip()
        positives = [
            "yes", "yeah", "add", "ok", "okay", "sure", "yep", "save",
            "ஆம்", "ஆமா", "சரி", "சேர்க்கலாம்", "ஆமாம்", "சேமி"
        ]
        if any(word in clean for word in positives):
            return "YES"
        return "NO"
