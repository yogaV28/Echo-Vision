import os
import sys
import time
import cv2
import requests
import logging
import re
import threading
import queue
import numpy as np

# Force PulseAudio for system sound routing
os.environ["SDL_AUDIODRIVER"] = "pulseaudio"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from voice_interface import EchoVoiceInterface
    from llm_assistant import EchoVisionAssistant
except ImportError as e:
    logging.error(f"[Orchestrator Boot] Failed to import modules: {e}")
    sys.exit(1)

# ----------------------------------------------------
# Hardware Network Targets (Raspberry Pi 5 LAN)
# ----------------------------------------------------
PI5_BASE_URL = "http://192.168.10.1:5000"
PI5_API_URL = f"{PI5_BASE_URL}/api/state"
REGISTER_API_URL = f"{PI5_BASE_URL}/api/register"
DECLINE_API_URL = f"{PI5_BASE_URL}/api/decline"
DEPTH_STREAM_URL = f"{PI5_BASE_URL}/video_feed/depth"

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
SYSTEM_LANGUAGE = "en"        # "en" for English, "ta" for Tamil
WAKE_WORD = "john"
UNKNOWN_DELAY_SECONDS = 90    # 1.5 minutes wait before unknown registration
KNOWN_COOLDOWN_SECONDS = 300  # 5 minutes cooldown before re-greeting

# 3D Depth Thresholds (in cm)
OBSTACLE_WARN_CM = 20         # Warning threshold
OBSTACLE_CRITICAL_CM = 10     # Critical stop threshold
OBSTACLE_COOLDOWN_SEC = 2.5   # Delay between consecutive obstacle alerts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global thread synchronization
speak_queue = queue.Queue()
interrupt_event = threading.Event()
mic_pause_event = threading.Event()
pending_id_lock = threading.Lock()
current_pending_id = None
last_obstacle_alert_time = 0


def speech_playback_worker(voice_interface):
    """Background Thread: Reads phrases from queue with instant interrupt support."""
    while True:
        text = speak_queue.get()
        if text is None:
            break
        
        phrases = re.split(r'[.,!?;\n]', text)
        phrases = [p.strip() for p in phrases if p.strip()]
        
        for phrase in phrases:
            if interrupt_event.is_set():
                logging.info("[Playback Interrupt] Active speech dropped.")
                break
            voice_interface.speak(phrase)
            
        speak_queue.task_done()
        if interrupt_event.is_set() and speak_queue.empty():
            interrupt_event.clear()


def force_immediate_interrupt():
    """Immediately stops AI speech playback and flushes the queue."""
    if not interrupt_event.is_set():
        logging.info("[System Interrupt] Flushing active audio buffer.")
        interrupt_event.set()
        while not speak_queue.empty():
            try:
                speak_queue.get_nowait()
                speak_queue.task_done()
            except queue.Empty:
                break


def depth_vision_worker(voice):
    """
    Background Thread: Processes the 3D Vision depth stream from the Pi 5.
    Calculates distances in Left, Center, and Right zones and generates navigation commands.
    """
    global last_obstacle_alert_time
    logging.info("[3D Vision Engine] Connecting to depth video feed...")
    
    cap = cv2.VideoCapture(DEPTH_STREAM_URL)
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(1.0)
            cap = cv2.VideoCapture(DEPTH_STREAM_URL)
            continue

        try:
            # Convert depth frame to single channel if received as BGR
            if len(frame.shape) == 3:
                depth_map = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                depth_map = frame

            h, w = depth_map.shape
            
            # Divide FOV into 3 sectors: Left, Center, Right
            left_zone = depth_map[:, :w // 3]
            center_zone = depth_map[:, w // 3: 2 * (w // 3)]
            right_zone = depth_map[:, 2 * (w // 3):]

            # Estimate distances in cm (scaled from depth intensities 0-255 -> 0-200cm range)
            def get_zone_distance(zone):
                valid_pixels = zone[zone > 10]
                if len(valid_pixels) == 0:
                    return 999.0
                # Higher brightness = closer object in typical normalized depth feeds
                closest_val = np.percentile(valid_pixels, 95)
                # Map 0-255 scale to ~ 0-200 cm
                dist_cm = max(5.0, (255.0 - closest_val) * (200.0 / 255.0))
                return dist_cm

            d_left = get_zone_distance(left_zone)
            d_center = get_zone_distance(center_zone)
            d_right = get_zone_distance(right_zone)
            
            min_dist = min(d_left, d_center, d_right)
            now = time.time()

            # --- OBSTACLE DETECTION & DIRECTIONAL GUIDANCE ---
            if min_dist <= OBSTACLE_WARN_CM and (now - last_obstacle_alert_time > OBSTACLE_COOLDOWN_SEC):
                last_obstacle_alert_time = now
                force_immediate_interrupt()
                interrupt_event.clear()

                # 1. Critical Hazard (<= 10 cm)
                if min_dist <= OBSTACLE_CRITICAL_CM:
                    if SYSTEM_LANGUAGE == "en":
                        warn_text = f"Warning! Obstacle at {int(min_dist)} centimeters! Stop immediately and step back."
                    else:
                        warn_text = f"எச்சரிக்கை! {int(min_dist)} சென்டிமீட்டரில் தடை! உடனே நின்று பின்வாங்கவும்."
                
                # 2. Navigation Guidance (10 cm to 20 cm)
                else:
                    if d_center <= OBSTACLE_WARN_CM:
                        if d_left > d_right and d_left > OBSTACLE_WARN_CM:
                            direction = "Move left." if SYSTEM_LANGUAGE == "en" else "இடது பக்கம் நகரவும்."
                        elif d_right > OBSTACLE_WARN_CM:
                            direction = "Move right." if SYSTEM_LANGUAGE == "en" else "வலது பக்கம் நகரவும்."
                        else:
                            direction = "Step back." if SYSTEM_LANGUAGE == "en" else "பின்வாங்கவும்."
                        
                        if SYSTEM_LANGUAGE == "en":
                            warn_text = f"Obstacle ahead at {int(d_center)} centimeters. {direction}"
                        else:
                            warn_text = f"முன்னால் {int(d_center)} செமீ தூரத்தில் தடை. {direction}"
                    
                    elif d_left <= OBSTACLE_WARN_CM:
                        warn_text = "Obstacle on your left. Move right." if SYSTEM_LANGUAGE == "en" else "இடதுபுறம் தடை. வலது பக்கம் நகரவும்."
                    
                    else: # d_right <= OBSTACLE_WARN_CM
                        warn_text = "Obstacle on your right. Move left." if SYSTEM_LANGUAGE == "en" else "வலதுபுறம் தடை. இடது பக்கம் நகரவும்."

                logging.warning(f"[3D Vision Alert] {warn_text}")
                speak_queue.put(warn_text)

        except Exception as e:
            logging.error(f"[3D Vision Error] Frame analysis error: {e}")

        time.sleep(0.08)  # ~12 FPS depth monitoring rate


def always_on_microphone_worker(voice, brain):
    """Background Thread: Continuous Gemini Live voice interaction."""
    logging.info("[Thread-Mic] Always-On Microphone Capture Active.")
    
    while True:
        if mic_pause_event.is_set():
            time.sleep(0.3)
            continue

        ambient_audio = voice.listen_for_keyword(timeout_seconds=3)
        
        if ambient_audio and not mic_pause_event.is_set():
            force_immediate_interrupt()
            interrupt_event.clear()
            
            clean_prompt = ambient_audio.lower().replace(WAKE_WORD, "").strip()
            if clean_prompt:
                logging.info(f"[User Query]: {clean_prompt}")
                try:
                    for chunk in brain.generate_narration_stream(clean_prompt):
                        if interrupt_event.is_set():
                            break
                        speak_queue.put(chunk)
                except AttributeError:
                    reply = brain.generate_narration(clean_prompt)
                    speak_queue.put(reply)


def start_echo_vision_loop():
    """Master Orchestrator Loop."""
    global current_pending_id
    logging.info("[System Run] Starting Master Core Orchestrator...")
    
    voice = EchoVoiceInterface(language=SYSTEM_LANGUAGE, model_size="tiny")
    brain = EchoVisionAssistant(language=SYSTEM_LANGUAGE, model_name="qwen2.5:3b")
    
    # Launch All Concurrent Workers
    threading.Thread(target=speech_playback_worker, args=(voice,), daemon=True).start()
    threading.Thread(target=always_on_microphone_worker, args=(voice, brain), daemon=True).start()
    threading.Thread(target=depth_vision_worker, args=(voice,), daemon=True).start()
    
    startup_msg = "John is ready with 3D Vision. Monitoring environment." if SYSTEM_LANGUAGE == "en" else "3D பார்வையுடன் ஜான் தயார். கண்காணிக்கப்படுகிறது."
    speak_queue.put(startup_msg)
    
    last_known_person = None
    last_known_time = 0
    last_processed_uid = None
    unknown_start_time = None
    has_prompted_for_current_unknown = False

    while True:
        try:
            response = requests.get(PI5_API_URL, timeout=1.0)
            if response.status_code == 200:
                telemetry = response.json()
                identified_list = telemetry.get("identified", [])
                pending_list = telemetry.get("pending", [])
                
                # --- 1. KNOWN FACE DETECTION ---
                if len(identified_list) > 0:
                    current_person = identified_list[0].get("name", "Unknown")
                    now = time.time()
                    
                    if current_person != last_known_person or (now - last_known_time > KNOWN_COOLDOWN_SECONDS):
                        logging.info(f"[Cam Event] Identified: {current_person}")
                        force_immediate_interrupt()
                        interrupt_event.clear()
                        
                        greeting_msg = (f"Hello {current_person}, welcome back." 
                                        if SYSTEM_LANGUAGE == "en" 
                                        else f"வணக்கம் {current_person}, மீண்டும் வருக.")
                        speak_queue.put(greeting_msg)
                        
                        last_known_person = current_person
                        last_known_time = now
                        unknown_start_time = None
                        has_prompted_for_current_unknown = False
                
                # --- 2. UNIDENTIFIED PERSON 1.5 MINUTE REGISTRATION ---
                elif len(pending_list) > 0:
                    pending_node = pending_list[0]
                    pending_id = pending_node.get("id")
                    pi5_reported_age = pending_node.get("age_sec", 0.0)
                    
                    with pending_id_lock:
                        current_pending_id = pending_id
                    
                    if pending_id != last_processed_uid:
                        last_processed_uid = pending_id
                        unknown_start_time = time.time()
                        has_prompted_for_current_unknown = False
                    
                    local_elapsed = time.time() - unknown_start_time if unknown_start_time else 0
                    total_elapsed = max(local_elapsed, pi5_reported_age)
                    
                    if total_elapsed >= UNKNOWN_DELAY_SECONDS and not has_prompted_for_current_unknown:
                        has_prompted_for_current_unknown = True
                        
                        mic_pause_event.set()
                        force_immediate_interrupt()
                        
                        prompt_msg = ("A new person is in front of you. Would you like to add this person?" 
                                      if SYSTEM_LANGUAGE == "en" 
                                      else "முன்னால் ஒரு புதிய நபர் உள்ளார். அவரை சேர்க்க வேண்டுமா?")
                        voice.speak(prompt_msg)
                        
                        confirmation = voice.listen_for_keyword(timeout_seconds=5)
                        parsed_ans = voice.parse_yes_no(confirmation)
                        
                        if parsed_ans == "YES":
                            name_prompt = "What is their name?" if SYSTEM_LANGUAGE == "en" else "அவரது பெயர் என்ன?"
                            voice.speak(name_prompt)
                            
                            raw_name = voice.listen_for_keyword(timeout_seconds=5)
                            clean_name = raw_name.strip().replace(" ", "_") if raw_name else ""
                            
                            if clean_name:
                                payload = {"pending_id": pending_id, "name": clean_name}
                                try:
                                    reg_res = requests.post(REGISTER_API_URL, json=payload, timeout=3.0)
                                    if reg_res.status_code == 200:
                                        success_msg = (f"{clean_name} has been added successfully." 
                                                       if SYSTEM_LANGUAGE == "en" 
                                                       else f"{clean_name} வெற்றிகரமாக சேர்க்கப்பட்டார்.")
                                        voice.speak(success_msg)
                                    else:
                                        voice.speak("Server registration error.")
                                except requests.RequestException:
                                    voice.speak("Connection error during registration.")
                            else:
                                requests.post(DECLINE_API_URL, json={"pending_id": pending_id}, timeout=2.0)
                                cancel_msg = "No name heard. Registration cancelled." if SYSTEM_LANGUAGE == "en" else "ரத்து செய்யப்பட்டது."
                                voice.speak(cancel_msg)
                        else:
                            try:
                                requests.post(DECLINE_API_URL, json={"pending_id": pending_id}, timeout=2.0)
                            except requests.RequestException:
                                pass
                            decline_msg = "Person declined." if SYSTEM_LANGUAGE == "en" else "நிராகரிக்கப்பட்டது."
                            voice.speak(decline_msg)
                        
                        mic_pause_event.clear()
                
                else:
                    last_processed_uid = None
                    unknown_start_time = None
                    has_prompted_for_current_unknown = False
                    with pending_id_lock:
                        current_pending_id = None

        except requests.exceptions.RequestException:
            pass

        time.sleep(0.2)


if __name__ == "__main__":
    start_echo_vision_loop()
