import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EchoVisionAssistant:
    def __init__(self, language="ta", model_name="qwen2.5:3b"): # Upgraded to 3B or 7B for smarter, Gemini-like reasoning
        self.language = language
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/chat"
        
        # Conversation history memory buffer
        self.conversation_history = []
        self.max_memory_turns = 12
        
        # Updated personality profile anchoring the name 'John'
        if self.language == "ta":
            self.system_prompt = (
                "நீ 'John' என்ற பெயருடைய ஒரு மிகவும் அன்பான, மென்மையான, மற்றும் அதிபுத்திசாலி AI உதவியாளர். "
                "உன் பயனர் பெயர் யோகா. எப்போதும் சுருக்கமாகவும், கனிவாகவும், தூய தமிழிலோ அல்லது "
                "தேவைப்பட்டால் ஆங்கிலம் கலந்த தங்கிலீஷிலோ தெளிவாகப் பேசவும். "
                "பயனர் உன்னைத் தடுக்கும்போது உடனடியாகப் பேச்சை நிறுத்திக்கொள்."
            )
        else:
            self.system_prompt = (
                "You are 'John', an incredibly warm, gentle, and state-of-the-art conversational AI assistant. "
                "Your user's name is Yoga. Always respond concisely, kindly, and intelligently. "
                "If the user interrupts or stops you, immediately yield and listen carefully."
            )

    def generate_narration(self, user_query):
        if not user_query:
            return ""

        logging.info(f"[LLM Brain] Processing query under model {self.model_name}: '{user_query}'")
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_query})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.5,   # Slightly lower temperature for balanced, empathetic, stable output
                "num_predict": 140    
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                assistant_reply = result.get("message", {}).get("content", "").strip()
                
                # Append to sliding dialogue memory window
                self.conversation_history.append({"role": "user", "content": user_query})
                self.conversation_history.append({"role": "assistant", "content": assistant_reply})
                
                if len(self.conversation_history) > self.max_memory_turns * 2:
                    self.conversation_history = self.conversation_history[-self.max_memory_turns * 2:]
                    
                return assistant_reply
            else:
                return "மன்னிக்கவும் யோகா, என் சேவையகத்தில் சிறிய பிழை." if self.language == "ta" else "I encountered a minor server error."
        except requests.exceptions.RequestException:
            return "மன்னிக்கவும் யோகா, என்னால் இப்போது பதிலளிக்க முடியவில்லை." if self.language == "ta" else "I cannot connect to my server right now."

    def process_event(self, event_type, context):
        """Generates real-time custom notification blocks for camera state changes."""
        name = context.get("name", "Unknown person")
        
        if event_type == "face_detected":
            if name == "Unidentified person":
                return (
                    "Excuse me, Yoga. A new unidentified person has arrived in front of us. "
                    "Would you like me to register them? If so, say: John, person name is [NAME], please add."
                ) if self.language == "en" else (
                    "மன்னிக்கவும் யோகா. ஒரு புதிய அறியப்படாத நபர் நம் முன்னால் வந்துள்ளார். "
                    "அவரைச் சேர்க்க விரும்பினால்: John அவர் பெயர் [NAME] சேர்க்கவும் என்று கூறவும்."
                )
            else:
                return (
                    f"Excuse me, Yoga. I see {name} has just walked in front of us. "
                    f"Let's greet them or let me know what you would like to do."
                ) if self.language == "en" else (
                    f"மன்னிக்கவும் யோகா. {name} நம் முன்னால் வந்துள்ளார் என்பதை நான் பார்க்கிறேன். "
                    f"அவரை வரவேற்போம், அல்லது நாம் என்ன செய்ய வேண்டும் என்று எனக்குக் கூறுங்கள்."
                )
        return ""