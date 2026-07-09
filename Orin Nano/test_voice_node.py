import sys
import os
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from voice_interface import EchoVoiceInterface
    from llm_assistant import EchoVisionAssistant
except ImportError as e:
    print(f"[ERROR] Failed to import dependency files: {e}")
    sys.exit(1)

def execute_voice_diagnostics():
    # Target Language Code: 'ta' for Tamil, 'en' for English
    TARGET_LANG = "en"
    
    print(f"[BOOT] Starting Echo-Vision Control Loop. Language context: {TARGET_LANG}")
    voice = EchoVoiceInterface(language=TARGET_LANG, model_size="tiny")
    brain = EchoVisionAssistant(language=TARGET_LANG, model_name="qwen2.5:1.5b")
    
    # 1. Simulate an event payload received from the Raspberry Pi 5
    mock_event = {"name": "Unidentified person"}
    
    # 2. Ask LLM to generate the alert question
    narration_script = brain.process_event("face_detected", mock_event)
    
    # 3. Speak the question to the user
    voice.speak(narration_script)
    
    # 4. Open the microphone stream and capture user response
    user_speech_capture = voice.listen_for_keyword(timeout_seconds=5)
    
    # 5. Parse the user's intent
    action_decision = voice.parse_yes_no(user_speech_capture)
    print(f"[DIAGNOSTIC] Tokenizer evaluation result: {action_decision}")
    
    # Contextual LLM verification if token analysis returns RETRY
    if action_decision == "RETRY" and user_speech_capture:
        print("[PROCESSING] Phrase fell outside strict array thresholds. Querying LLM for contextual intent matching...")
        intent_prompt = (
            f"Analyze this speech phrase: '{user_speech_capture}'. "
            "Does the user mean YES or NO to saving the person? Reply with only one word: YES or NO."
        )
        llm_verdict = brain.generate_narration(intent_prompt)
        if "YES" in llm_verdict.upper():
            action_decision = "YES"
        elif "NO" in llm_verdict.upper():
            action_decision = "NO"

    # 6. Take action based on the verified choice
    if action_decision == "YES":
        followup_prompt = "பெயரைக் கூறவும்" if TARGET_LANG == "ta" else "Please state the name clearly."
        voice.speak(followup_prompt)
        
        captured_name = voice.listen_for_keyword(timeout_seconds=4)
        success_prompt = f"{captured_name} வெற்றிகரமாக பதிவு செய்யப்பட்டார்." if TARGET_LANG == "ta" else f"Successfully stored template for {captured_name}."
        voice.speak(success_prompt)
    else:
        abort_prompt = "செயல்பாடு ரத்து செய்யப்பட்டது." if TARGET_LANG == "ta" else "Action ignored by user."
        voice.speak(abort_prompt)

if __name__ == "__main__":
    execute_voice_diagnostics()