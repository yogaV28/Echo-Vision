"""
Local face database.

Layout on disk:
    face_db/
        embeddings.json          # {name: [[512 floats], ...]}  (<=6 per name)
        <person_name>/
            001.jpg
            002.jpg
            ...                  # <=6 images, oldest dropped first

Only local storage is used -- nothing leaves the Pi 5 in this module.
"""

import json
import os
import time
import cv2
import numpy as np

import config


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


class FaceDatabase:
    def __init__(self, root: str = config.DB_ROOT):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.embeddings_path = os.path.join(self.root, "embeddings.json")
        self.people: dict[str, list[np.ndarray]] = {}
        self._load()

    # ---- persistence ----------------------------------------------------
    def _load(self):
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "r") as f:
                raw = json.load(f)
            self.people = {name: [np.array(e, dtype=np.float32) for e in embs]
                            for name, embs in raw.items()}
        else:
            self.people = {}

    def _save(self):
        raw = {name: [e.tolist() for e in embs] for name, embs in self.people.items()}
        with open(self.embeddings_path, "w") as f:
            json.dump(raw, f)

    def _person_dir(self, name: str) -> str:
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        return d

    # ---- writes -----------------------------------------------------------
    def add_face(self, name: str, embedding: np.ndarray, face_crop_bgr: np.ndarray):
        """Add one (image, embedding) pair for `name`, enforcing the 6-image cap."""
        person_dir = self._person_dir(name)
        existing = sorted(
            f for f in os.listdir(person_dir) if f.lower().endswith(".jpg")
        )

        embs = self.people.setdefault(name, [])

        # FIFO rotation once at the cap. Renumber the remaining files
        # contiguously (001, 002, ...) so the next filename can never
        # collide with (and silently overwrite) a file still in use --
        # that collision previously let embeddings and stored images
        # drift out of sync.
        if len(existing) >= config.MAX_IMAGES_PER_PERSON:
            os.remove(os.path.join(person_dir, existing[0]))
            existing = existing[1:]
            if embs:
                embs.pop(0)

            renumbered = []
            for idx, fname in enumerate(existing, start=1):
                new_name = f"{idx:03d}.jpg"
                if fname != new_name:
                    os.rename(os.path.join(person_dir, fname), os.path.join(person_dir, new_name))
                renumbered.append(new_name)
            existing = renumbered

        next_index = len(existing) + 1
        filename = f"{next_index:03d}.jpg"
        cv2.imwrite(os.path.join(person_dir, filename), face_crop_bgr)

        embs.append(embedding.astype(np.float32))
        self._save()

    def person_count(self, name: str) -> int:
        return len(self.people.get(name, []))

    def known_names(self) -> list[str]:
        return list(self.people.keys())

    # ---- reads --------------------------------------------------------------
    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (best_name, best_score) or (None, best_score) if below threshold."""
        best_name, best_score = None, -1.0
        for name, embs in self.people.items():
            for e in embs:
                score = cosine_similarity(embedding, e)
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_name is not None and best_score >= config.MATCH_THRESHOLD:
            return best_name, best_score
        return None, best_score


class PendingUnidentified:
    """
    Tracks faces that were flagged 'Unidentified' so we don't re-prompt the
    user about the same face every single frame. Purely in-memory.
    """

    def __init__(self):
        self._entries: list[dict] = []  # {embedding, last_asked}

    def should_prompt(self, embedding: np.ndarray) -> bool:
        now = time.time()
        for entry in self._entries:
            sim = cosine_similarity(embedding, entry["embedding"])
            if sim >= config.PENDING_SIMILARITY:
                if now - entry["last_asked"] < config.UNIDENTIFIED_COOLDOWN_SEC:
                    return False
                entry["last_asked"] = now
                return True
        # New unseen unidentified face
        self._entries.append({"embedding": embedding, "last_asked": now})
        return True

    def clear(self, embedding: np.ndarray):
        """Call after a face is successfully registered, so it stops being 'pending'."""
        self._entries = [
            e for e in self._entries
            if cosine_similarity(embedding, e["embedding"]) < config.PENDING_SIMILARITY
        ]