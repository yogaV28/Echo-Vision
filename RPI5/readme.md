# 🍓 Echo-Vision: Raspberry Pi 5 Sensor & API Hub

This repository contains the Raspberry Pi 5 edge node code for the Echo-Vision project. It is responsible for interfacing with the Night Camera and 3D Vision sensor, maintaining the local face database, running lightweight inference, and serving real-time video streams and API endpoints over the LAN to the Jetson Orin Nano.

## 📂 File Structure & Functions

Based on the repository architecture, here is the breakdown of the RPi 5 modules:

*   **`face_db/`** (Directory): Stores registered face image data and local vector embeddings for known users.
*   **`app.py`**: The main entry point that initializes the camera feeds, state manager, and web server.
*   **`camera.py`**: Hardware interfacing script for managing the dual cameras (Night Vision via V4L2/libcamera and the 3D Depth Sensor).
*   **`config.py`**: Centralized configuration variables (LAN IP, port mappings, depth thresholds).
*   **`database.py`**: Manages local SQLite/storage operations for enrolled user profiles.
*   **`face_engine.py`**: Core logic wrapping the face detection and recognition models to extract and compare facial features.
*   **`inference_worker.py`**: A background worker thread dedicated to running face inference asynchronously, ensuring the camera streams never block or lag.
*   **`requirements.txt`**: The Python pip dependency manifest for the Pi 5.
*   **`rpi5_streamer.py`**: Handles the MJPEG video streaming of the camera feeds to the Orin Nano.
*   **`shared_state.py`**: A thread-safe state manager that tracks newly detected "pending" faces versus "identified" known persons across threads.
*   **`web_server.py`**: The Flask server that exposes the endpoints (`/api/state`, `/api/register`, `/api/decline`) consumed by the Orin Nano orchestrator.

---

## ⚠️ Caution: System Dependencies for Raspberry Pi 5

The Raspberry Pi 5 uses Raspberry Pi OS "Bookworm". This OS uses a strict Python environment and relies on `libcamera` for native hardware. To get OpenCV and camera streams working, you must install the underlying Linux libraries before setting up Python.

Run these commands in your Pi 5 terminal before installing your Python requirements:

```bash
# 1. Update the system
sudo apt-get update

# 2. Install OpenCV hardware dependencies and V4L2 utilities
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev v4l-utils

# 3. Optional: Install native picamera2 library if using the official Ribbon cable camera
sudo apt install -y python3-picamera2
```

---

## 🛠️ How to Install the Python Requirements

Because Raspberry Pi OS Bookworm protects the global system Python environment, you **must** use a virtual environment. Furthermore, to access system packages like `picamera2`, you must pass the `--system-site-packages` flag.

```bash
# 1. Create a virtual environment named "echo_env" with access to system packages
python3 -m venv --system-site-packages echo_env

# 2. Activate the virtual environment
source echo_env/bin/activate

# 3. Install the requirements
pip install -r requirements.txt
```

---

## 🚀 Running the Node

1. Connect both the Night Camera and the 3D Vision sensor to the USB/CSI ports on the Pi 5.
2. Ensure the Pi is connected to the Orin Nano via the LAN cable with a static IP (e.g., `192.168.10.1`).
3. Activate the virtual environment and start the master app:
```bash
source echo_env/bin/activate
python3 app.py
```
4. The API will now be live on port `5000`, and the Orin Nano will automatically connect to pull telemetry and depth feeds.
