import cv2
import numpy as np
import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RPI5_IP = "192.168.10.1"
NIGHT_STREAM_URL = f"http://{RPI5_IP}:5000/video_feed/night"
DEPTH_STREAM_URL = f"http://{RPI5_IP}:5000/video_feed/depth"

def process_streams():
    logging.info("[Orin Processor] Connecting to RPI5 camera streams over LAN...")
    
    night_stream = cv2.VideoCapture(NIGHT_STREAM_URL)
    depth_stream = cv2.VideoCapture(DEPTH_STREAM_URL)

    while True:
        ret_n, night_frame = night_stream.read()
        ret_d, depth_frame = depth_stream.read()

        if not ret_n or not ret_d:
            logging.warning("[Stream Error] Lost connection to RPI5 streams. Retrying...")
            time.sleep(1.0)
            try:
                night_stream = cv2.VideoCapture(NIGHT_STREAM_URL)
                depth_stream = cv2.VideoCapture(DEPTH_STREAM_URL)
            except:
                pass
            continue

        # --- ORIN GPU PROCESSING ---
        
        # 1. Face Detection & Recognition on Night Camera frame
        # (Offload heavy model inference like YOLO / FaceNet onto Orin Nano CUDA)
        gray = cv2.cvtColor(night_frame, cv2.COLOR_BGR2GRAY)
        # Placeholder for local face detection processing...

        # 2. Distance and Obstacle Estimation on 3D Vision Depth frame
        # Convert depth frame to distance metrics
        depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        min_distance = np.min(depth_gray[depth_gray > 0]) if np.any(depth_gray > 0) else 0.0
        
        if min_distance < 50:  # Threshold for close obstacle alert (cm/units)
            logging.warning(f"[Obstacle Alert] Object detected at close range: {min_distance} units!")

        # Display frames locally on Orin GUI if needed
        cv2.imshow("Orin - Night Camera (Face Processing)", night_frame)
        cv2.imshow("Orin - 3D Vision (Obstacle Mapping)", depth_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_streams()
