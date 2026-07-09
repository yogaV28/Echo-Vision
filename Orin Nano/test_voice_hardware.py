import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from voice_interface import EchoVoiceInterface
except ImportError as e:
    print("[ERROR] Could not find voice_interface.py! Make sure this script is in the same folder.")
    print(e)
    sys.exit(1)

def run_voice_suite():
    print("==============================================")
    print("      ECHO-VISION SYSTEM VOICE DIAGNOSTICS    ")
    print("==============================================\n")
    
    # PHASE 1: TEST ENGLISH PIPELINE
    print("[PHASE 1] Initializing Voice Interface in English Mode...")
    voice_en = EchoVoiceInterface(language="en", model_size="tiny")
    
    print("\n--> Testing Text-to-Speech (TTS)...")
    voice_en.speak("Hello Yoga. Initiating voice verification profile. Please listen carefully.")
    time.sleep(1)
    
    print("\n--> Testing Speech-to-Text (STT) + Token Parsing...")
    voice_en.speak("Should I save this person to the local database? Please speak now.")
    
    user_audio_en = voice_en.listen_for_keyword(timeout_seconds=5)
    decision_en = voice_en.parse_yes_no(user_audio_en)
    
    print(f"\n[RAW TRANSCRIPTION]: '{user_audio_en}'")
    print(f"[PARSED DECISION]: {decision_en}")
    
    if decision_en == "YES":
        voice_en.speak("Action confirmed. Registration sequence started.")
    elif decision_en == "NO":
        voice_en.speak("Action denied. Clearing temporary image buffer.")
    else:
        voice_en.speak("Input not understood. Defaulting to standby mode.")
        
    print("\n----------------------------------------------")
    time.sleep(2)

    # PHASE 2: TEST TAMIL PIPELINE
    print("[PHASE 2] Switching Language Context to Tamil Mode...")
    voice_ta = EchoVoiceInterface(language="ta", model_size="tiny")
    
    print("\n--> தமிழ் உரை-ஒலி மாற்ற சோதனை (TTS)...")
    voice_ta.speak("வணக்கம் யோகா. குரல் பதிவு சோதனையைத் தொடங்குகிறேன்.")
    time.sleep(1)
    
    print("\n--> தமிழ் ஒலி-உரை மாற்ற சோதனை (STT) + டோக்கன் பகுப்பாய்வு...")
    voice_ta.speak("புதிய நபரை அமைப்பில் சேர்க்க வேண்டுமா? ஆம் அல்லது இல்லை என்று கூறவும்.")
    
    user_audio_ta = voice_ta.listen_for_keyword(timeout_seconds=5)
    decision_ta = voice_ta.parse_yes_no(user_audio_ta)
    
    print(f"\n[மூலப் பிரதி (RAW TRANSCRIPTION)]: '{user_audio_ta}'")
    print(f"[முடிவு (PARSED DECISION)]: {decision_ta}")
    
    if decision_ta == "YES":
        voice_ta.speak("சரி, நபரின் பெயரைப் பதிவு செய்யவும்.")
    elif decision_ta == "NO":
        voice_ta.speak("தரவு நிராகரிக்கப்பட்டது.")
    else:
        voice_ta.speak("மன்னிக்கவும், உரை புரியவில்லை.")

    print("\n==============================================")
    print("           DIAGNOSTIC TEST COMPLETE           ")
    print("==============================================")

if __name__ == "__main__":
    run_voice_suite()