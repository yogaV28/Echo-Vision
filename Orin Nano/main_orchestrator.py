import os
import sys
import time
import requests
import logging
import re
import threading
import queue

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from voice_interface import EchoVoiceInterface
    from llm_assistant import EchoVisionAssistant
except ImportError as e:
    logging.error(f"[Orchestrator Boot] Failed to import dependency modules: {e}")
    sys.exit(1)

# Hardware Network Targets
PI5_BASE_URL = "http://192.168.10.1:5000"
PI5_API_URL = f"{PI5_BASE_URL}/api/state"  
REGISTER_API_URL = f"{PI5_BASE_URL}/api/register"
DECLINE_API_URL = f"{PI5_BASE_URL}/api/decline"

SYSTEM_LANGUAGE = "en"  # 'ta' for Tamil, 'en' for English
WAKE_WORD = "john"       # Name changed cleanly to 'John'
UNKNOWN_COOLDOWN_DELAY = 60  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables for real-time thread synchronization
speak_queue = queue.Queue()
interrupt_event = threading.Event()
pending_id_lock = threading.Lock()
current_pending_id = None

def speech_playback_worker(voice_interface):
    """Background Thread: Asynchronously reads out phrases, responding instantly to interrupts."""
    while True:
        text = speak_queue.get()
        if text is None:
            break
        
        # Segment sentences into atomic phrases for micro-cadence delivery
        phrases = re.split(r'[.,!?;\n]', text)
        phrases = [p.strip() for p in phrases if p.strip()]
        
        for phrase in phrases:
            if interrupt_event.is_set():
                logging.info("[Playback Interrupt] Dropping active speech block processing.")
                break
            voice_interface.speak(phrase)
            
        speak_queue.task_done()
        interrupt_event.clear()

def extract_name_from_speech(phrase):
    phrase_clean = phrase.lower().strip()
    en_matches = re.findall(r"person name is\s+([a-zA-Z0-9_\s]+?)(?:\s+please\s+add|$)", phrase_clean)
    if en_matches:
        return en_matches[0].strip().replace(" ", "_")
    ta_matches = re.findall(r"பெயர்\s+([a-zA-Z0-9_\s\u0B80-\u0BFF]+?)(?:\s+சேர்க்கவும்|$)", phrase_clean)
    if ta_matches:
        return ta_matches[0].strip().replace(" ", "_")
    return None

def force_immediate_interrupt():
    """Forces the system to immediately drop current audio tasks to execute high-priority events."""
    if not interrupt_event.is_set():
        logging.info("[System Interrupt] Triggering immediate audio buffer flush.")
        interrupt_event.set()
        while not speak_queue.empty():
            try:
                speak_queue.get_nowait()
                speak_queue.task_done()
            except queue.Empty:
                break

def always_on_microphone_worker(voice, brain):
    """Background Thread: Continuously captures microphone input and processes conversational replies."""
    global current_pending_id
    logging.info("[Thread-Mic] Always-On Microphone Capture Pipeline Hot.")
    
    while True:
        # Record voice input slices
        ambient_audio = voice.listen_for_keyword(timeout_seconds=3)
        
        if ambient_audio:
            # Turn-Taking Cadence Check: If user voice is detected, drop active AI playback immediately
            force_immediate_interrupt()
            
            if WAKE_WORD in ambient_audio.lower():
                logging.info(f"[Wake Event] John captured request stream: '{ambient_audio}'")
                
                # A. Check for conversational database registration command override
                captured_name = extract_name_from_speech(ambient_audio)
                with pending_id_lock:
                    active_uid = current_pending_id

                if captured_name and active_uid:
                    logging.info(f"[Enrollment] Mapping name '{captured_name}' to ID: {active_uid}")
                    payload = {"pending_id": active_uid, "name": captured_name}
                    try:
                        res = requests.post(REGISTER_API_URL, json=payload, timeout=3)
                        if res.status_code == 200:
                            success_msg = f"Successfully registered {captured_name}." if SYSTEM_LANGUAGE == "en" else f"{captured_name} வெற்றிகரமாக சேர்க்கப்பட்டார்."
                            speak_queue.put(success_msg)
                            with pending_id_lock:
                                current_pending_id = None
                        else:
                            speak_queue.put("Server update error.")
                    except requests.exceptions.RequestException:
                        speak_queue.put("Connection failed.")
                    continue

                # B. Execute standard generative conversational assistant task
                clean_query = ambient_audio.lower().replace(WAKE_WORD, "").strip()
                if clean_query:
                    reply = brain.generate_narration(clean_query)
                    speak_queue.put(reply)

def start_echo_vision_loop():
    global current_pending_id
    logging.info("[System Run] Launching Master Central Core Orchestrator...")
    
    # Update model initialization target to 3b or 7b for complex logic processing
    voice = EchoVoiceInterface(language=SYSTEM_LANGUAGE, model_size="tiny")
    brain = EchoVisionAssistant(language=SYSTEM_LANGUAGE, model_name="qwen2.5:3b")
    
    # Deploy asynchronous workers
    threading.Thread(target=speech_playback_worker, args=(voice,), daemon=True).start()
    threading.Thread(target=always_on_microphone_worker, args=(voice, brain), daemon=True).start()
    
    init_msg = "John active. Monitoring environment." if SYSTEM_LANGUAGE == "en" else "ஜான் செயல்பாட்டில் உள்ளார். கண்காணிக்கப்படுகிறது."
    speak_queue.put(init_msg)
    
    last_processed_uid = None
    last_processed_id = None
    unknown_start_time = None
    has_prompted_for_current_unknown = False
    last_state = None

    while True:
        try:
            response = requests.get(PI5_API_URL, timeout=1)
            if response.status_code == 200:
                telemetry = response.json()
                current_state = telemetry.get("state", "idle")
                
                identified_list = telemetry.get("identified", [])
                pending_list = telemetry.get("pending", [])
                
                # --- CASE A: KNOWN PERSON ENTERS FOV (INTERRUPT PREVIOUS CONVERSATION NATIVELY) ---
                if len(identified_list) > 0:
                    current_person = identified_list[0].get("name", "Unknown")
                    if current_person != last_processed_id:
                        logging.info(f"[LAN Cam Event] High Priority: Known person seen -> {current_person}")
                        
                        # Stop whatever the AI was saying previously to prioritize this update
                        force_immediate_interrupt()
                        
                        # Process contextual notification text via the custom event engine
                        alert_text = brain.process_event("face_detected", {"name": current_person})
                        speak_queue.put(alert_text)
                        
                        last_processed_id = current_person
                        unknown_start_time = None
                        has_prompted_for_current_unknown = False
                
                # --- CASE B: UNIDENTIFIED PERSON SPOTTED (INTERRUPT AT 1-MINUTE MARK) ---
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
                        logging.info(f"[LAN Cam Event] Unregistered tracker active. Token: {pending_id}")
                    
                    local_elapsed = time.time() - unknown_start_time if unknown_start_time else 0
                    max_elapsed_age = max(local_elapsed, pi5_reported_age)
                    
                    if max_elapsed_age >= UNKNOWN_COOLDOWN_DELAY and not has_prompted_for_current_unknown:
                        logging.info(f"[Trigger Alert] High Priority: 1 Minute limit hit for unknown face. Interrupting...")
                        
                        # Force current voice workflows down to immediately inject the registration prompt
                        force_immediate_interrupt()
                        
                        alert_text = brain.process_event("face_detected", {"name": "Unidentified person"})
                        speak_queue.put(alert_text)
                        has_prompted_for_current_unknown = True
                
                # --- CASE C: STANDBY SYSTEM IDLE ---
                elif current_state == "idle" or (len(identified_list) == 0 and len(pending_list) == 0):
                    if last_state != "idle":
                        logging.info("[LAN Cam Event] Field clear. System returns to standby.")
                        last_processed_id = None
                        last_processed_uid = None
                        with pending_id_lock:
                            current_pending_id = None
                        unknown_start_time = None
                        has_prompted_for_current_unknown = False

                last_state = current_state
                
        except requests.exceptions.RequestException:
            pass

        time.sleep(0.1)

if __name__ == "__main__":
    start_echo_vision_loop()