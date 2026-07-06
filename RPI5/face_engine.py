"""
Wraps MTCNN (detection) + InceptionResnetV1 (embedding) from facenet-pytorch.

detect() returns only faces whose bounding box falls inside the configured
range-gate ratios (see config.MIN_FACE_HEIGHT_RATIO / MAX_FACE_HEIGHT_RATIO),
which approximates "only act on people in the 10-15m band" from a 2D camera.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

import config


@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray
    crop_bgr: np.ndarray


class FaceEngine:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # keep_all=True so we can see every face in frame, then range-gate ourselves
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def get_boxes(self, frame_bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """Detect face boxes only (no embedding) -- cheap, safe to run on a
        downscaled frame. Returns [((x1,y1,x2,y2), box_h_ratio), ...]."""
        frame_h = frame_bgr.shape[0]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None:
            return []

        results = []
        for box, prob in zip(boxes, probs):
            if prob is None or prob < 0.90:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            box_h_ratio = (y2 - y1) / frame_h
            if not (config.MIN_FACE_HEIGHT_RATIO <= box_h_ratio <= config.MAX_FACE_HEIGHT_RATIO):
                continue  # outside the configured range band
            results.append(((x1, y1, x2, y2), box_h_ratio))
        return results

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        return self._embed(crop_bgr)

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Convenience one-shot: boxes + embeddings from the SAME frame.
        Simple but more expensive than the get_boxes()+embed() split used by
        the threaded web pipeline -- kept for the legacy console app."""
        results = []
        for box, _ratio in self.get_boxes(frame_bgr):
            x1, y1, x2, y2 = box
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            embedding = self.embed(crop_bgr)
            if embedding is None:
                continue
            results.append(DetectedFace(box=box, embedding=embedding, crop_bgr=crop_bgr))
        return results

    def _embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            face_tensor = self.mtcnn.extract(rgb, [[0, 0, rgb.shape[1], rgb.shape[0]]], None)
            if face_tensor is None:
                # fall back to simple resize if extract() (alignment) fails
                resized = cv2.resize(rgb, (160, 160))
                face_tensor = torch.tensor(resized).permute(2, 0, 1).float()
                face_tensor = (face_tensor - 127.5) / 128.0
                face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            with torch.no_grad():
                emb = self.resnet(face_tensor)
            return emb[0].cpu().numpy()
        except Exception:
            return None