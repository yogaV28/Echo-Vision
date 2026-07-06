"""
Echo-Vision - Face ID module, web version.

Run:
    python3 app.py

Then open http://<pi5-ip>:5000/ in a browser on the same LAN, or have the
Orin Nano poll http://<pi5-ip>:5000/api/state directly.

Three things run concurrently, each in its own thread, so nothing blocks
the live video:
  1. Camera capture      (camera.py)       - always grabs the newest frame
  2. Inference worker     (inference_worker.py) - detects/matches, skips
     frames and downscales for speed, never blocks on human decisions
  3. Flask web server     (web_server.py)  - serves the MJPEG stream + JSON
     API; runs on the main thread

See web_server.py's module docstring for the exact API contract the Orin
Nano should use.
"""

import config
from camera import Camera
from database import FaceDatabase, PendingUnidentified
from face_engine import FaceEngine
from inference_worker import InferenceWorker
from shared_state import SharedState
from web_server import create_app


def main():
    print("Starting Echo-Vision face ID web service...")
    camera = Camera()
    engine = FaceEngine()
    db = FaceDatabase()
    pending_tracker = PendingUnidentified()
    state = SharedState()

    worker = InferenceWorker(camera, engine, db, pending_tracker, state)
    worker.start()

    app = create_app(camera, db, state)
    print(f"Known people in DB: {db.known_names()}")
    print(f"Serving on http://{config.WEB_HOST}:{config.WEB_PORT}  "
          f"(reachable from the Orin Nano over the LAN cable)")

    try:
        # threaded=True: Flask handles the video stream and API requests
        # concurrently instead of queuing them behind each other.
        app.run(host=config.WEB_HOST, port=config.WEB_PORT, threaded=True, use_reloader=False)
    finally:
        worker.stop()
        camera.release()


if __name__ == "__main__":
    main()