"""
Flask web server for the face ID module.

Routes:
  GET  /                render a simple live-view page (browser testing)
  GET  /video_feed       MJPEG stream, always shows the latest camera frame
                          with the latest detection boxes overlaid -- never
                          blocked by how fast detection is running
  GET  /api/state        JSON: current identified people + pending
                          unidentified people + known people list
  POST /api/register     {"pending_id": "...", "name": "..."} -> adds the
                          person to the local DB and starts opportunistic
                          multi-image enrollment for a few seconds
  POST /api/decline      {"pending_id": "..."} -> discards that pending
                          entry (cooldown already applied by should_prompt)

This is the exact boundary the Orin Nano (or its LLM/mic layer) talks to
over the LAN cable: poll /api/state, then POST a decision to /api/register
or /api/decline once the user answers "yes"/"no" by voice.
"""

import time

import cv2
from flask import Flask, Response, jsonify, render_template, request

import config
from camera import Camera
from database import FaceDatabase
from shared_state import SharedState

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Echo-Vision - Face ID</title>
  <style>
    body { font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }
    h1 { font-size: 18px; }
    #layout { display: flex; gap: 24px; flex-wrap: wrap; }
    img#stream { max-width: 720px; width: 100%; border: 1px solid #444; border-radius: 6px; }
    .panel { min-width: 280px; }
    .card { background: #1c1c1c; border: 1px solid #333; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .card img { width: 80px; border-radius: 4px; display: block; margin-bottom: 8px; }
    button { padding: 6px 12px; margin-right: 6px; border-radius: 4px; border: none; cursor: pointer; }
    .yes { background: #2e7d32; color: white; }
    .no { background: #c62828; color: white; }
    input[type=text] { padding: 6px; border-radius: 4px; border: 1px solid #555; background: #222; color: #eee; margin-bottom: 6px; width: 90%; }
    ul { padding-left: 18px; }
    code { color: #9fd; }
  </style>
</head>
<body>
  <h1>Echo-Vision - Face ID (live, LAN-served)</h1>
  <div id="layout">
    <img id="stream" src="/video_feed">
    <div class="panel">
      <div class="card">
        <strong>Known people</strong>
        <ul id="known"></ul>
      </div>
      <div class="card">
        <strong>Unidentified people awaiting a decision</strong>
        <div id="pending">None right now.</div>
      </div>
      <div class="card">
        <strong>API for the Orin Nano</strong>
        <p>GET <code>/api/state</code> &middot; POST <code>/api/register</code> &middot; POST <code>/api/decline</code></p>
      </div>
    </div>
  </div>

<script>
let pendingCards = {}; // id -> {el, ageEl, nameInput}

async function refresh() {
  const res = await fetch('/api/state');
  const data = await res.json();

  document.getElementById('known').innerHTML =
    data.known_people.length
      ? data.known_people.map(n => `<li>${n}</li>`).join('')
      : '<li><em>none yet</em></li>';

  const pendingDiv = document.getElementById('pending');
  const seenIds = new Set(data.pending.map(p => p.id));

  // Remove cards for people no longer pending (registered/declined elsewhere)
  for (const id of Object.keys(pendingCards)) {
    if (!seenIds.has(id)) {
      pendingCards[id].el.remove();
      delete pendingCards[id];
    }
  }

  if (!data.pending.length) {
    if (!Object.keys(pendingCards).length) pendingDiv.innerHTML = 'None right now.';
    return;
  }
  if (pendingDiv.textContent === 'None right now.') pendingDiv.innerHTML = '';

  for (const p of data.pending) {
    if (pendingCards[p.id]) {
      // Existing card: only refresh the timestamp, never touch the input
      // (that's what was stealing focus / wiping typed text every second).
      pendingCards[p.id].ageEl.textContent = `Seen ${p.age_sec}s ago`;
      continue;
    }

    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <img src="data:image/jpeg;base64,${p.thumbnail_jpeg_base64}">
      <div class="age">Seen ${p.age_sec}s ago</div>
      <input type="text" placeholder="Name">
      <div>
        <button class="yes">Yes, add</button>
        <button class="no">No</button>
      </div>
    `;
    const nameInput = card.querySelector('input');
    const ageEl = card.querySelector('.age');
    card.querySelector('.yes').onclick = () => register(p.id, nameInput);
    card.querySelector('.no').onclick = () => decline(p.id);
    pendingDiv.appendChild(card);
    pendingCards[p.id] = { el: card, ageEl, nameInput };
  }
}

async function register(id, inputEl) {
  const name = inputEl.value.trim();
  if (!name) { alert('Enter a name first'); return; }
  await fetch('/api/register', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pending_id: id, name})
  });
  if (pendingCards[id]) { pendingCards[id].el.remove(); delete pendingCards[id]; }
  refresh();
}

async function decline(id) {
  await fetch('/api/decline', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pending_id: id})
  });
  if (pendingCards[id]) { pendingCards[id].el.remove(); delete pendingCards[id]; }
  refresh();
}

setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""


def _draw_boxes(frame, boxes):
    colors = {
        "identified": (0, 200, 0),
        "unidentified": (0, 0, 255),
        "enrolling": (0, 165, 255),
    }
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        color = colors.get(b.get("status"), (200, 200, 200))
        label = b["label"]
        if "score" in b:
            label = f"{label} ({b['score']:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def create_app(camera: Camera, db: FaceDatabase, state: SharedState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/video_feed")
    def video_feed():
        def generate():
            min_interval = 1.0 / config.STREAM_FPS_LIMIT
            while True:
                start = time.time()
                ok, frame = camera.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                frame = _draw_boxes(frame, state.get_boxes())
                ok, buf = cv2.imencode(
                    ".jpg", frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_JPEG_QUALITY],
                )
                if not ok:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                elapsed = time.time() - start
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/state")
    def api_state():
        boxes = state.get_boxes()
        identified = [b for b in boxes if b.get("status") == "identified"]
        enrollment = state.get_enrollment()
        return jsonify({
            "identified": [
                {"name": b["label"], "score": b["score"], "box": b["box"]}
                for b in identified
            ],
            "pending": state.get_pending_public(),
            "known_people": db.known_names(),
            "active_enrollment": enrollment["name"] if enrollment else None,
            "timestamp": time.time(),
        })

    @app.route("/api/register", methods=["POST"])
    def api_register():
        data = request.get_json(force=True, silent=True) or {}
        pending_id = data.get("pending_id")
        name = "".join(
            c for c in str(data.get("name", "")).strip()
            if c.isalnum() or c in ("_", "-", " ")
        ).strip().replace(" ", "_")

        if not pending_id or not name:
            return jsonify({"ok": False, "error": "pending_id and name are required"}), 400

        entry = state.pop_pending(pending_id)
        if entry is None:
            return jsonify({"ok": False, "error": "unknown or expired pending_id"}), 404

        db.add_face(name, entry["embedding"], entry["crop_bgr"])
        state.start_enrollment(
            name=name,
            embedding=entry["embedding"],
            target_count=config.MAX_IMAGES_PER_PERSON,
            window_sec=config.ENROLLMENT_WINDOW_SEC,
        )
        return jsonify({"ok": True, "name": name, "images_saved": db.person_count(name)})

    @app.route("/api/decline", methods=["POST"])
    def api_decline():
        data = request.get_json(force=True, silent=True) or {}
        pending_id = data.get("pending_id")
        if not pending_id:
            return jsonify({"ok": False, "error": "pending_id is required"}), 400
        entry = state.pop_pending(pending_id)
        return jsonify({"ok": entry is not None})

    return app