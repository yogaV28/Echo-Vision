"""
Thread-safe state shared between:
  - the inference worker thread (writes detections/pending people)
  - the Flask request threads (read state, write register/decline decisions)

Nothing here talks to the camera or the model directly -- it's just a
mailbox, which keeps the threading model simple to reason about.
"""

import base64
import threading
import time
import uuid

import cv2
import numpy as np


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.boxes: list[dict] = []          # current frame's drawable boxes
        self.pending: dict[str, dict] = {}    # pending_id -> {box, embedding, crop_bgr, thumbnail_b64, created_at}
        self.active_enrollment: dict | None = None  # {name, embedding, deadline, count}
        self.known_people: list[str] = []

    # ---- detections (worker writes, web reads) ----------------------------
    def set_boxes(self, boxes: list[dict]):
        with self._lock:
            self.boxes = boxes

    def get_boxes(self) -> list[dict]:
        with self._lock:
            return list(self.boxes)

    def set_known_people(self, names: list[str]):
        with self._lock:
            self.known_people = names

    # ---- pending unidentified people ---------------------------------------
    def add_pending(self, box, embedding: np.ndarray, crop_bgr: np.ndarray) -> str:
        pending_id = uuid.uuid4().hex[:8]
        ok, buf = cv2.imencode(".jpg", crop_bgr)
        thumbnail_b64 = base64.b64encode(buf).decode("ascii") if ok else ""
        with self._lock:
            self.pending[pending_id] = {
                "box": box,
                "embedding": embedding,
                "crop_bgr": crop_bgr,
                "thumbnail_b64": thumbnail_b64,
                "created_at": time.time(),
            }
        return pending_id

    def pop_pending(self, pending_id: str) -> dict | None:
        with self._lock:
            return self.pending.pop(pending_id, None)

    def get_pending_public(self) -> list[dict]:
        """JSON-safe view (no raw embedding/crop arrays)."""
        with self._lock:
            return [
                {
                    "id": pid,
                    "box": entry["box"],
                    "thumbnail_jpeg_base64": entry["thumbnail_b64"],
                    "age_sec": round(time.time() - entry["created_at"], 1),
                }
                for pid, entry in self.pending.items()
            ]

    # ---- active enrollment (auto-capture extra images after a "yes") ------
    def start_enrollment(self, name: str, embedding: np.ndarray, target_count: int, window_sec: float):
        with self._lock:
            self.active_enrollment = {
                "name": name,
                "embedding": embedding,
                "deadline": time.time() + window_sec,
                "target_count": target_count,
            }

    def get_enrollment(self) -> dict | None:
        with self._lock:
            if self.active_enrollment is None:
                return None
            if time.time() > self.active_enrollment["deadline"]:
                self.active_enrollment = None
                return None
            return dict(self.active_enrollment)

    def clear_enrollment(self):
        with self._lock:
            self.active_enrollment = None