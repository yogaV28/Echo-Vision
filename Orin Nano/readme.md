# 👁️ Echo-Vision: Orin Nano Core

Echo-Vision is an advanced, local AI personal assistant built for the NVIDIA Jetson Orin Nano. It acts as the "Central Brain," bridging real-time 3D spatial awareness, facial recognition, and a "Gemini-Live" style conversational voice interface routed directly to Bluetooth smart glasses (like the pTron Orbis Neo).

## ✨ Key Features

* **Gemini-Live Style Interaction:** Utilizes an always-on microphone with Voice Activity Detection (VAD). You can instantly interrupt the AI while it is speaking, and it will immediately stop to listen to your new command.
* **3D Vision & Obstacle Avoidance:** Analyzes depth streams to calculate distances in Left, Center, and Right zones. Provides real-time navigational guidance for obstacles within 20 cm and issues critical emergency stop warnings for objects under 10 cm.
* **Smart Face Enrollment:** Polls a Raspberry Pi 5 over a LAN connection for face tracking. If an unidentified person is in frame for 1.5 minutes, the AI initiates an interactive voice enrollment sequence to capture their name and register them in the database.
* **Bilingual & Adaptive Learning:** Supports both English (`en`) and Tamil (`ta`). Features a Dynamic RAG Memory System that detects Tamil interactions, logs them, and injects them into the LLM's context window so the AI naturally learns your specific phrasing and dialect over time.
* **System-Level Audio Routing:** Bypasses ALSA hardware conflicts by piping high-quality Google TTS (`gTTS`) and `espeak-ng` directly through PulseAudio to your connected Bluetooth glasses.

---

## 📂 Code Structure & Functions

* **`main_orchestrator.py`**  
  The master script. It runs concurrent background threads for speech playback, always-on microphone listening (VAD), 3D depth stream processing, and Pi 5 API polling. It handles the 1.5-minute registration logic and immediate obstacle interruption.
* **`voice_interface.py`**  
  The audio engine. Handles microphone capture, runs `faster-whisper` on the GPU for instant Speech-to-Text, and pipes Google TTS / `espeak-ng` directly to the system's default PulseAudio sink.
* **`llm_assistant.py`**  
  The Local AI brain. Interfaces with the local LLM (`qwen2.5:3b`), handles prompt generation, and manages short-term contextual memory injection for Tamil learning.
* **`orin_processor3D.py`**  
  A dedicated vision processor script designed to handle 3D depth camera mapping and local face processing workloads.
* **`client.py` / `test_voice_hardware.py` / `test_voice_node.py`**  
  Diagnostic and testing utilities for hardware microphones, audio sinks, and network endpoint verification.
* **`requirements.txt`**  
  The unified dependency manifest containing PyTorch, OpenCV, Whisper, and Audio libraries.

---

## ⚙️ Installation & Setup

### ⚠️ Crucial Step: System Dependencies for Jetson
Because the Jetson Orin Nano runs an ARM-based Linux environment, audio packages (like `PyAudio`) and voice engines require system hardware drivers before Python packages can be installed.

Run this command in your Orin Nano terminal before installing the Python packages:

```bash
# Install Linux audio drivers, OpenCV dependencies, and TTS engines
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio mpg123 espeak-ng libcanberra-gtk-module libcanberra-gtk3-module
```

### Install Python Requirements

Once system drivers are configured, activate your virtual environment and install the dependencies:

```bash
# Install the Python packages
pip install -r requirements.txt
```

> **Note:** `opencv-python` handles the `cv2` imports for processing the 3D depth stream, `pyaudio` manages microphone capture, and `mpg123` provides low-latency playback of Google TTS audio streams directly to your Bluetooth headset.

---

## 🚀 Running the System

1. Ensure your Raspberry Pi 5 is powered on, connected via LAN (`192.168.10.1`), and hosting the camera feeds.
2. Connect your Bluetooth glasses to the Jetson Orin Nano. Open `pavucontrol` in the terminal to verify the glasses profile is set to **Headset (HSP/HFP)**.
3. Launch the orchestrator:
```bash
python3 main_orchestrator.py
```
4. If testing audio routing for the first time, open `pavucontrol` and ensure the `mpg123` / `espeak-ng` stream is assigned to your Bluetooth device.
