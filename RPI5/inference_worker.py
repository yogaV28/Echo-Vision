"""
Background inference thread.

Runs independently of the camera capture thread and the Flask request
threads. It:
  1. Skips most frames (config.DETECT_EVERY_N_FRAMES) -- detection is the
     expensive part, the video stream doesn't need it done every frame.
  2. Detects on a downscaled copy of the frame, then scales boxes back up
     and crops the FULL-resolution region for embedding, so recognition
     quality isn't hurt by the speed optimization.
  3. Never blocks waiting for a human decision. When an unidentified face
     is seen, it drops a "pending" entry into SharedState and moves on --
     the actual yes/no/name decision arrives later via the web/API and is
     applied on a future loop iteration (see start_enrollment()).
"""

import threading
import time

import cv2

import config
from camera import Camera
from database import FaceDatabase, PendingUnidentified, cosine_similarity
from face_engine import FaceEngine
from shared_state import SharedState


class InferenceWorker:
    def __init__(self, camera: Camera, engine: FaceEngine, db: FaceDatabase,
                 pending_tracker: PendingUnidentified, state: SharedState):
        self.camera = camera
        self.engine = engine
        self.db = db
        self.pending_tracker = pending_tracker
        self.state = state
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self):
        frame_count = 0
        scale = config.DETECTION_DOWNSCALE

        while self._running:
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % config.DETECT_EVERY_N_FRAMES != 0:
                time.sleep(0.005)
                continue

            small = cv2.resize(frame, None, fx=scale, fy=scale)
            raw_boxes = self.engine.get_boxes(small)

            enrollment = self.state.get_enrollment()
            display_boxes = []

            for (sx1, sy1, sx2, sy2), _ratio in raw_boxes:
                x1, y1 = int(sx1 / scale), int(sy1 / scale)
                x2, y2 = int(sx2 / scale), int(sy2 / scale)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                embedding = self.engine.embed(crop)
                if embedding is None:
                    continue

                # --- opportunistic multi-image capture after a "yes" -----
                if enrollment is not None:
                    sim = cosine_similarity(embedding, enrollment["embedding"])
                    already = self.db.person_count(enrollment["name"])
                    if sim >= config.PENDING_SIMILARITY and already < enrollment["target_count"]:
                        self.db.add_face(enrollment["name"], embedding, crop)
                        display_boxes.append({
                            "box": [x1, y1, x2, y2],
                            "label": f"Adding {enrollment['name']}... "
                                     f"({already + 1}/{enrollment['target_count']})",
                            "status": "enrolling",
                        })
                        if already + 1 >= enrollment["target_count"]:
                            self.state.clear_enrollment()
                        continue

                # --- normal identify / flag-as-unidentified ---------------
                name, score = self.db.match(embedding)
                if name is not None:
                    display_boxes.append({
                        "box": [x1, y1, x2, y2],
                        "label": name,
                        "score": round(score, 3),
                        "status": "identified",
                    })
                else:
                    display_boxes.append({
                        "box": [x1, y1, x2, y2],
                        "label": "Unidentified person",
                        "status": "unidentified",
                    })
                    if self.pending_tracker.should_prompt(embedding):
                        self.state.add_pending((x1, y1, x2, y2), embedding, crop)

            self.state.set_boxes(display_boxes)
            self.state.set_known_people(self.db.known_names())