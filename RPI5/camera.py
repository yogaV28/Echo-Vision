"""
Camera abstraction, threaded so capture never blocks on detection/network work.

Two things fixed here vs. a naive OpenCV loop:

1. COLOR BUG: Picamera2's "RGB888" format is a confusingly-named libcamera
   quirk -- despite the name, it actually hands you bytes in BGR order
   (matching OpenCV's convention). Converting it with cv2.COLOR_RGB2BGR (as
   an earlier version of this file did) flips it a SECOND time, which is
   exactly what produced the blue/purple-tinted preview.

2. LAG: capture now runs in its own background thread that grabs frames as
   fast as the camera can produce them into a single "latest frame" slot.
   Anything downstream (the web video stream, the detection worker) just
   reads whatever's newest instead of waiting in line behind slow work.
"""

import threading
import time

import cv2
import numpy as np

import config


class Camera:
    """Continuously-updated camera source. .read() never blocks and always
    returns the most recent frame available."""

    def __init__(self):
        self.backend = config.CAMERA_BACKEND
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        if self.backend == "picamera2":
            self._init_picamera2()
        elif self.backend == "opencv":
            self._init_opencv()
        else:
            raise ValueError(f"Unknown CAMERA_BACKEND: {self.backend}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _init_picamera2(self):
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "picamera2 is not installed. On Raspberry Pi OS it's "
                "usually preinstalled system-wide; if you're in a venv, "
                "recreate it with --system-site-packages, or run "
                "'sudo apt install -y python3-picamera2'."
            ) from e

        self._picam = Picamera2()
        cfg = self._picam.create_preview_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "RGB888"}
        )
        self._picam.configure(cfg)
        self._picam.start()

    def _init_opencv(self):
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Could not open camera with OpenCV. Check config.CAMERA_INDEX, "
                "or switch config.CAMERA_BACKEND to 'picamera2' if this is "
                "the CSI ribbon-cable camera."
            )

    def _grab_raw(self) -> np.ndarray | None:
        if self.backend == "picamera2":
            # NOTE: "RGB888" from picamera2 is already BGR byte order.
            # Do NOT run cv2.cvtColor(..., COLOR_RGB2BGR) on this.
            frame = self._picam.capture_array()
        else:
            ok, frame = self._cap.read()
            frame = frame if ok else None

        if frame is not None and config.CAMERA_ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        return frame

    def _capture_loop(self):
        while self._running:
            frame = self._grab_raw()
            if frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Non-blocking: returns the most recently captured frame."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.backend == "picamera2":
            self._picam.stop()
        else:
            self._cap.release()