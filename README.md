<div align="center">
  <img src="[https://img.icons8.com/fluency/96/000000/artificial-intelligence.png](https://img.icons8.com/fluency/96/000000/artificial-intelligence.png)" alt="Echo-Vision Icon" width="100"/>
  <h1>👁️ Echo-Vision</h1>
  <p><strong>Advanced Local AI Assistant & 3D Spatial Awareness System</strong></p>
</div>

---

Echo-Vision is a multi-node AI system utilizing an NVIDIA Jetson Orin Nano (Central Brain) and a Raspberry Pi 5 (Edge Sensor Hub). It integrates local Large Language Models (LLMs), real-time 3D depth mapping, facial recognition, and a bilingual conversational voice interface (English/Tamil) routed to Bluetooth smart glasses.

## 🗺️ Project Map

Here is the full module breakdown and current completion status for Echo-Vision's architecture:

- [x] **1. Face detection + ID (Pi 5)** — MTCNN + local DB, max 6 photos/person, range-gated to 10–15m (Built, including the extra LAN-server variant).
- [x] **2. LAN bridge / web API (Pi 5 ↔ Orin Nano)** — HTTP endpoints so the Orin can poll pending "unidentified person" events and post back yes/no + name.
- [x] **3. Voice interface (mic + speaker, Tamil/English)** — Speech-to-text for answering "add this person?" and giving a name; text-to-speech for all system announcements.
- [x] **4. Local LLM assistant (Orin Nano)** — The conversational layer that ties everything together: announces identifications, relays questions, answers general queries, speaks in Tamil or English.
- [x] **5. LiDAR 3D room mapping (Orin Nano + YDLidar sensor)** — Builds a live depth/occupancy picture of the room, 30–40m outer radius, feeds distances to modules 6 and 7.
- [x] **6. Object detection + proximity warning** — YOLO-based detection fused with LiDAR depth; triggers "object is near, please [direction]" under 5m.
- [ ] **7. Directional guidance algorithm** — Turns the LiDAR's clear-space map into concrete left/right/back/forward instructions (the "brain" behind module 6's warnings).
- [ ] **8. Main orchestrator (Orin Nano)** — The top-level loop tying modules 2–7 together: decides what the LLM says, arbitrates between "someone's here" vs "obstacle nearby", and manages system state.

---

## 🌐 LAN Code Setup Commands

To establish a direct communication bridge between the Orin Nano and the Raspberry Pi 5, follow these networking steps:

Check your active interfaces:
```bash
ip link show
```

**On the Orin Nano:**
```bash
sudo nmcli connection add type ethernet ifname enP8p1s0 con-name echo-vision-lan ip4 192.168.10.2/24
sudo nmcli connection up echo-vision-lan
```

**On the Pi 5:**
```bash
sudo nmcli connection add type ethernet ifname eth0 con-name echo-vision-lan ip4 192.168.10.1/24
sudo nmcli connection up echo-vision-lan
```

**Verify Connection:**
From the Orin Nano:
```bash
ping 192.168.10.1
```
From the Pi 5:
```bash
ping 192.168.10.2
```

**Jetson Orin Nano Client Setup and Running:**
Test the API bridge using the client script or cURL:
```bash
python3 client.py 192.168.10.1
curl -v http://192.168.10.1:5000/
```

---

## 🧠 Local LLM Setup & Troubleshooting

### For LLM Loading
Open a new terminal tab on your Orin Nano and start the Ollama service manually:
```bash
ollama serve
```
Leave that terminal window running in the background. Open another terminal tab and run:
```bash
ollama run qwen2.5:1.5b
```

### Issue: LLM Crashing / Freezing (Force CPU Mode)
If the model struggles with GPU allocation, tell the Ollama background service to disable CUDA acceleration and run inference entirely via the CPU threads.

**Step 1: Edit the Service**
```bash
sudo systemctl edit ollama.service
```
Add these exact lines to the file:
```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="CUDA_VISIBLE_DEVICES=-1"
```
Save the file and exit the editor (if using Nano, press `Ctrl+O`, `Enter`, then `Ctrl+X`).

**Step 2: Reload and Restart Ollama**
Apply your newly modified configuration overrides and restart the background daemon:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```
Verify that the model boots cleanly on the CPU without crashing:
```bash
ollama run qwen2.5:1.5b
```
*(If the terminal prompt appears, type a quick test message, then type `/exit` to close it).*

### Resolving Persistent `AttributeError` (VS Code Cache Issue)
If Python still throws an `AttributeError` for custom functions (like `parse_yes_no`), VS Code is likely auto-saving into the wrong location or your virtual environment is caching an old file. Force a direct copy:
```bash
cp /home/ieee/Documents/Echo-Vision/voice_interface.py /home/ieee/Documents/Echo-Vision/.venv/lib/python3.10/site-packages/ 2>/dev/null || true
```
Now, run your integration script:
```bash
python3 test_voice_node.py
```
The system will now execute seamlessly, process your voice prompt, translate your choice via the token parser, and output speech.

---

## 🎧 Bluetooth Setup & Troubleshooting

Ah, the classic "No Bluetooth Found" issue on Linux. Since the Jetson Orin Nano Developer Kit does not come with an on-board Wi-Fi/Bluetooth module on the M.2 slot out of the box, you must configure external hardware correctly.

### Step 1: Check if the Hardware is Present
```bash
hciconfig -a
```
*   If it returns completely blank: You either don't have a Bluetooth module plugged in, or the kernel module isn't loading.
*   If using a USB Dongle: Run `lsusb` and look for a line containing "Bluetooth Radio" or a manufacturer name (e.g., Realtek, CSR).

### Step 2: Restart the Bluetooth Linux Daemon
Sometimes the system service crashes during heavy package installations. Force-restart it:
```bash
sudo systemctl restart bluetooth
sudo systemctl enable bluetooth
sudo systemctl status bluetooth
```
*(Look for `Active: active (running)` in green text).*

### Step 3: Unblock Radio Devices (RFKILL)
Check if a soft-lock is blocking wireless transmitters:
```bash
rfkill list
```
If you see "Soft blocked: yes", force it to unblock:
```bash
sudo rfkill unblock bluetooth
```

### Step 4: Clear the Audio Connection and Complete the Link
Restart the desktop user-space audio daemon so it fetches the clean hardware variables:
```bash
pulseaudio -k
pulseaudio --start
```
Return to `bluetoothctl` and establish the link with your glasses/headset:
```bash
sudo bluetoothctl
```
Inside the prompt, connect using your device's MAC address:
```bash
connect 39:88:B7:32:59:8B
# OR
connect 41:42:FF:A3:56:14
```

### 🛠️ Hardware Solution (If you lack a Bluetooth module)
If you haven't plugged a module into the Jetson yet, you have two quick options:
1.  **USB Bluetooth 5.0/5.3 Dongle:** The fastest option. Plug a standard USB adapter (like a TP-Link UB500) into an open USB port.
2.  **M.2 Key E Card:** Install an Intel AX200/AX210 wireless card into the small slot underneath the Orin Nano heatsink and attach the antenna leads.
