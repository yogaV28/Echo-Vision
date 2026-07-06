"""
Echo-Vision face ID module - configuration.

All tunable thresholds live here so they can be calibrated on-site without
touching pipeline logic.
"""

import os

# ---- Storage ----------------------------------------------------------
DB_ROOT = os.path.join(os.path.dirname(__file__), "face_db")
EMBEDDINGS_FILE = os.path.join(DB_ROOT, "embeddings.json")
MAX_IMAGES_PER_PERSON = 6  # oldest image is dropped (FIFO) once exceeded

# ---- Recognition --------------------------------------------------------
# Cosine similarity threshold for a positive match. Higher = stricter.
# facenet-pytorch (vggface2 weights) embeddings: 0.6-0.7 is a reasonable
# starting point. Tune using known-good/known-bad pairs from your own data.
MATCH_THRESHOLD = 0.65

# ---- Range gating (requirement #3: only act within ~10-15m) -----------
# A 2D camera has no native depth, so range is approximated by how large
# the detected face is in the frame: a face far away produces a small
# bounding box. This ratio = face_box_height / frame_height.
# Lower bound: face must be at least this big -> i.e. person must be
# closer than the outer range limit.
# You MUST calibrate these two numbers for your actual lens (6mm CCTV lens
# per your hardware) by standing at 10m and 15m and reading off box heights.
MIN_FACE_HEIGHT_RATIO = 0.03   # ~ outer edge of range (person near 15m)
MAX_FACE_HEIGHT_RATIO = 0.90   # ignore absurdly close/occluding faces

# ---- New-person prompt behaviour ---------------------------------------
# How long (seconds) to wait before re-asking about a person who was
# already declined ("No"), so the system doesn't nag every frame.
UNIDENTIFIED_COOLDOWN_SEC = 30
# Cosine similarity used purely to recognise "this is the same unidentified
# face I already asked about", separate from the stricter MATCH_THRESHOLD.
PENDING_SIMILARITY = 0.75

# ---- Camera --------------------------------------------------------------
# "picamera2" -> for the CSI ribbon-cable camera (your setup) via libcamera.
#                Required on Raspberry Pi OS Bookworm; cv2.VideoCapture()
#                generally cannot read CSI cameras there.
# "opencv"    -> for a USB webcam (cv2.VideoCapture works fine for those).
CAMERA_BACKEND = "picamera2"
CAMERA_INDEX = 0  # only used when CAMERA_BACKEND == "opencv"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Set True if the camera module is physically mounted upside-down (common
# with CSI ribbon cables routed a certain way) so the image comes out
# inverted. Applied once at capture time, so everything downstream (face
# detection, the web stream, saved images) already sees it right-side up.
CAMERA_ROTATE_180 = True

# ---- Registration --------------------------------------------------------
# How many extra frames to capture (in addition to the triggering frame)
# when a new person is confirmed, up to MAX_IMAGES_PER_PERSON total.
CAPTURE_FRAMES_ON_REGISTER = 5

# ---- Performance (threaded web pipeline) ---------------------------------
# Detection is the expensive step. Two knobs cut its cost without touching
# accuracy of a positive ID:
#   - Only run detection on every Nth captured frame.
#   - Run it on a downscaled copy of the frame (boxes are scaled back up).
# The live video stream itself is NOT limited by these -- it always shows
# the newest camera frame, so the picture stays smooth even if detection
# is slower than the camera's frame rate.
DETECT_EVERY_N_FRAMES = 3
DETECTION_DOWNSCALE = 0.5   # 0.5 = run MTCNN on a half-resolution copy
STREAM_JPEG_QUALITY = 80
STREAM_FPS_LIMIT = 15       # cap on how many frames/sec the browser gets

# ---- Web server / LAN API --------------------------------------------------
# The Orin Nano (or a browser) reaches this over the LAN cable.
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
# How long (seconds) after a registration the worker keeps opportunistically
# grabbing extra images of the same face to fill the 6-image quota.
ENROLLMENT_WINDOW_SEC = 8