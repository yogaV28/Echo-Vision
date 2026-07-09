import os
import sys
import queue
import logging
import pyttsx3
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EchoVoiceInterface:
    def __init__(self, language="en", model_size="tiny"):
        self.language = language  
        self.audio_queue = queue.Queue()
        
        try:
            self.tts_engine = pyttsx3.init()
            self._configure_tts_voice()
            self.hardware_audio_output = True
        except Exception as e:
            logging.error(f"Failed to initialize hardware TTS backend: {e}")
            self.hardware_audio_output = False

        try:
            logging.info(f"Loading Whisper model '{model_size}' onto CPU...")
            self.stt_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.hardware_mic_input = True
        except Exception as e:
            logging.error(f"Failed to initialize STT model or Mic audio: {e}")
            self.hardware_mic_input = False

    def _configure_tts_voice(self):
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('rate', 160)  
        
        if self.language == "ta":
            for voice in voices:
                if "tamil" in voice.name.lower() or "ta" in voice.languages:
                    self.tts_engine.setProperty('voice', voice.id)
                    return
        else:
            for voice in voices:
                if "english" in voice.name.lower() or "en" in voice.languages:
                    self.tts_engine.setProperty('voice', voice.id)
                    return

    def change_language(self, language_code):
        if language_code in ["en", "ta"]:
            self.language = language_code
            self._configure_tts_voice()

    def speak(self, text):
        if not text:
            return
        print(f"\n[Echo-Vision Output]: {text}")
        if self.hardware_audio_output:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS execution error: {e}")

    def listen_for_keyword(self, timeout_seconds=3):
        if not self.hardware_mic_input:
            return input("[Console Mic Input]: ").strip()

        p = pyaudio.PyAudio()
        chunk_size = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 16000  
        
        # --- FIXED PULSEAUDIO ENFORCEMENT LOOP ---
        input_device_index = None
        try:
            # Force target to system default index directly if Pulse is running
            default_device = p.get_default_input_device_info()
            input_device_index = default_device.get('index')
            logging.info(f"[Mic Select] Successfully bound directly to System Default Pulse/PulseAudio stream (Index {input_device_index})")
        except Exception:
            # Fallback search if system defaults are masked by NoMachine environment
            info = p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount', 0)
            for i in range(0, numdevices):
                try:
                    device_info = p.get_device_info_by_host_api_device_index(0, i)
                    dev_name = device_info.get('name', '').lower()
                    if device_info.get('maxInputChannels', 0) > 0:
                        if "pulse" in dev_name or "default" in dev_name or "nomachine" in dev_name:
                            input_device_index = i
                            logging.info(f"[Mic Select] Found valid virtual host stream: {dev_name} at index {i}")
                            break
                except Exception:
                    continue

        try:
            stream = p.open(format=sample_format, channels=channels, rate=fs,
                            input_device_index=input_device_index,
                            frames_per_buffer=chunk_size, input=True)
        except Exception as e:
            logging.error(f"[Mic Failure] Could not open audio stream bridge: {e}")
            p.terminate()
            return ""

        print("[Listening... Speak now]")
        frames = []
        for _ in range(0, int(fs / chunk_size * timeout_seconds)):
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                break

        print("[Processing speech...]")
        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_bytes = b"".join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            segments, info = self.stt_model.transcribe(audio_np, beam_size=1, language=self.language)
            transcription = " ".join([segment.text for segment in segments])
            # Filter out tiny whisper hallucinations caused by dead mic background static
            if len(transcription.strip()) <= 2:
                return ""
            print(f"[Transcribed Text]: {transcription}")
            return transcription.strip()
        except Exception:
            return ""

    def parse_yes_no(self, phrase):
        if not phrase:
            return "RETRY"
        phrase_clean = phrase.lower().strip()
        
        en_yes = ["yes", "yeah", "add him", "add her", "save", "okay", "sure"]
        en_no = ["no", "dont", "don't", "leave", "ignore", "skip"]
        ta_yes = ["ஆம்", "ஆமாம்", "ஆமா", "சேமி", "சரி", "சேர்க்கலாம்", "ஆமாங்"]
        ta_no = ["இல்லை", "வேண்டாம்", "வேணாம்", "விட்டுவிடு", "நிராகரி", "இல்ல"]

        if any(word in phrase_clean for word in en_yes) or any(word in phrase_clean for word in ta_yes):
            return "YES"
        if any(word in phrase_clean for word in en_no) or any(word in phrase_clean for word in ta_no):
            return "NO"
        return "RETRY"