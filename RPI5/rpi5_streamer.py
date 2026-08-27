import cv2
import flask
from flask import Response
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = flask.Flask(__name__)

# Camera indices or paths (adjust based on your v4l2 device nodes)
# Night camera typically /dev/video0 or /dev/video2, 3D Vision depth stream on another index
NIGHT_CAM_INDEX = 0
DEPTH_CAM_INDEX = 2  

night_cap = cv2.VideoCapture(NIGHT_CAM_INDEX)
depth_cap = cv2.VideoCapture(DEPTH_CAM_INDEX)

night_frame = None
depth_frame = None
lock = threading.Lock()

def capture_loops():
    global night_frame, depth_frame
    while True:
        success_night, n_frame = night_cap.read()
        success_depth, d_frame = depth_cap.read()
        
        with lock:
            if success_night:
                night_frame = n_frame
            if success_depth:
                depth_frame = d_frame

def generate_mjpeg(camera_type):
    global night_frame, depth_frame
    while True:
        with lock:
            frame = night_frame if camera_type == "night" else depth_frame
            if frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed/night")
def night_feed():
    return Response(generate_mjpeg("night"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed/depth")
def depth_feed():
    return Response(generate_mjpeg("depth"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/telemetry")
def telemetry():
    # Return basic status or hardware vitals from RPI5 to Orin
    return {"status": "active", "rpi_temp": 45.0}

if __name__ == "__main__":
    t = threading.Thread(target=capture_loops, daemon=True)
    t.start()
    # Bind to the LAN interface IP assigned to the Pi
    app.run(host="0.0.0.0", port=5000, threaded=True)
