#!/usr/bin/env python3
"""
hand.py -- Two-hand webcam gesture controller for Asphalt 8.

Two fists in front of the webcam act as a virtual steering wheel (A/D).
Independently of that, either open hand drifts/brakes (S), and a middle-
finger gesture on either hand fires nitro (SPACE). Each hand is read on its
own, so "one hand steers, the other hand drifts" works naturally.

IMPLEMENTATION NOTE: this uses MediaPipe's current Tasks API
(mediapipe.tasks.vision.HandLandmarker), not the old mp.solutions.hands
class that most older tutorials show -- that class was removed in
MediaPipe 1.0. The Tasks API needs a small model file (hand_landmarker.task),
which this script downloads automatically next to itself the first time it
runs (see ensure_model_downloaded()).

Controls
--------
  Q  quit (releases every key first)
  R  recalibrate the steering center

See the accompanying write-up for setup, a gesture explanation, and
troubleshooting.
"""

import os
import sys
import time
import math
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from pynput.keyboard import Controller as PynputController, Key


# ============================================================
# CONFIGURATION -- tune these to fit your camera, hands and taste
# ============================================================

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# --- MediaPipe model ---
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6        # confidence required to *find* a hand
MIN_HAND_PRESENCE_CONFIDENCE = 0.5    # confidence that a tracked hand is still there
MIN_TRACKING_CONFIDENCE = 0.5         # confidence required to keep tracking a hand

# --- Steering ---
STEERING_DEAD_ZONE = 8.0     # degrees of relative angle treated as "centered"
STEERING_HYSTERESIS = 3.0    # degrees of "give" before a steering key is released again
STEERING_SENSITIVITY = 1.0   # angle multiplier; >1 = touchier, <1 = needs more rotation
ANGLE_SMOOTHING = 0.35       # EMA alpha for the steering angle, 0-1 (higher = snappier)
INVERT_STEERING = False      # flip to True if left/right ever feel backwards

# --- Calibration ---
CALIBRATION_COUNTDOWN = 3.0     # seconds of "get ready" countdown
CALIBRATION_SAMPLE_TIME = 2.5   # seconds spent averaging the "straight ahead" pose

# --- Gesture stability / debounce (section 13 of the brief: anti-flicker) ---
GESTURE_STABLE_FRAMES = 4    # consecutive frames a gesture must hold before it's trusted
SPACE_COOLDOWN = 0.6         # minimum seconds between separate SPACE activations

# --- Finger-extended thresholds (tune if detection is too strict/loose) ---
FINGER_EXTENDED_RATIO = 1.15   # how much farther than its PIP joint a fingertip must be from the wrist
THUMB_EXTENDED_RATIO = 0.55    # thumb-tip-to-index-knuckle distance, as a fraction of palm size

# --- Misc ---
WINDOW_NAME = "Asphalt 8 Hand Controller"
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

LEFT_COLOR = (255, 140, 0)    # BGR
RIGHT_COLOR = (0, 200, 255)   # BGR


# ============================================================
# SMALL GENERIC HELPERS
# ============================================================

def _dist(p1, p2):
    """Euclidean distance between two landmarks in normalized (x, y) space."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def put_text(frame, text, pos, color=(255, 255, 255), scale=0.55, thickness=1):
    """cv2.putText with a black outline so it stays readable over any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ============================================================
# HAND GEOMETRY / GESTURE RECOGNITION
#
# MediaPipe's 21 hand landmarks, for reference:
#   0 wrist
#   1-4   thumb  (CMC, MCP, IP, TIP)
#   5-8   index  (MCP, PIP, DIP, TIP)
#   9-12  middle (MCP, PIP, DIP, TIP)
#   13-16 ring   (MCP, PIP, DIP, TIP)
#   17-20 pinky  (MCP, PIP, DIP, TIP)
# ============================================================

def get_finger_states(landmarks):
    """
    Returns {'thumb'/'index'/'middle'/'ring'/'pinky': is_extended (bool)}.

    Uses distance-from-wrist ratios rather than a plain y-coordinate check,
    so it keeps working while a hand is tilted for steering rather than only
    when it's held perfectly upright.
    """
    wrist = landmarks[0]
    palm_ref = _dist(wrist, landmarks[9]) + 1e-6  # wrist -> middle knuckle, a size reference

    states = {}
    joints = {
        'index':  (6, 8),   # (PIP, TIP)
        'middle': (10, 12),
        'ring':   (14, 16),
        'pinky':  (18, 20),
    }
    for name, (pip_i, tip_i) in joints.items():
        pip = landmarks[pip_i]
        tip = landmarks[tip_i]
        states[name] = _dist(wrist, tip) > _dist(wrist, pip) * FINGER_EXTENDED_RATIO

    # The thumb moves mostly sideways rather than up/down, so it gets its own
    # check: how far the tip has swung away from the index knuckle.
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    states['thumb'] = _dist(thumb_tip, index_mcp) > THUMB_EXTENDED_RATIO * palm_ref

    return states


def detect_fist(states):
    """A fist: index/middle/ring/pinky are all folded. Thumb position is
    ignored on purpose -- it varies a lot between people and doesn't matter
    for steering."""
    return not states['index'] and not states['middle'] and not states['ring'] and not states['pinky']


def detect_open_palm(states):
    """An open palm: all five fingers, including the thumb, are extended."""
    return states['thumb'] and states['index'] and states['middle'] and states['ring'] and states['pinky']


def detect_middle_finger(states):
    """Only the middle finger is extended; index/ring/pinky are folded."""
    return states['middle'] and not states['index'] and not states['ring'] and not states['pinky']


def classify_gesture(states):
    """
    One confirmed gesture label for a single hand's finger states, checked in
    priority order MIDDLE > OPEN_PALM > FIST > UNKNOWN (spec section 12).
    """
    if detect_middle_finger(states):
        return 'MIDDLE'
    if detect_open_palm(states):
        return 'OPEN_PALM'
    if detect_fist(states):
        return 'FIST'
    return 'UNKNOWN'


def get_fist_center(landmarks, frame_w, frame_h):
    """Pixel-space center of a hand: the average of the wrist and the four
    finger-base knuckles. Stable whether the hand is a fist or not."""
    idxs = (0, 5, 9, 13, 17)
    cx = sum(landmarks[i].x for i in idxs) / len(idxs)
    cy = sum(landmarks[i].y for i in idxs) / len(idxs)
    return (int(cx * frame_w), int(cy * frame_h))


def get_hand_orientation_angle(landmarks, frame_w, frame_h):
    """
    A single hand's own tilt: the angle (degrees, image coordinates) of the
    vector from the wrist to the middle-finger knuckle. Used as the fallback
    steering signal when only one fist is available (spec section 11).
    """
    wrist = landmarks[0]
    mcp = landmarks[9]
    dx = (mcp.x - wrist.x) * frame_w
    dy = (mcp.y - wrist.y) * frame_h
    return math.degrees(math.atan2(dy, dx))


def calculate_steering_angle(left_center, right_center):
    """
    Angle (degrees) of the line from the left fist to the right fist, in
    OpenCV's image coordinate system (x right, y down).

    Convention: right hand lower than left hand -> positive angle -> steer
    RIGHT. Right hand higher than left hand -> negative angle -> steer LEFT.
    Set INVERT_STEERING = True if this feels backwards on your setup.
    """
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    return math.degrees(math.atan2(dy, dx))


def normalize_angle(angle):
    """Wraps an angle into (-180, 180]. A safety net -- hands never actually
    rotate this far, but the calibration math shouldn't break if they did."""
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def smooth_angle(previous, new_value, alpha):
    """Exponential moving average. `previous` may be None for the first sample."""
    if previous is None:
        return new_value
    return alpha * new_value + (1.0 - alpha) * previous


# ============================================================
# STABILITY / DEBOUNCE / STATE-MACHINE HELPERS
# (plain functions + small dicts -- no classes needed here)
# ============================================================

def new_gesture_stability_state():
    return {'confirmed': 'NOT DETECTED', 'candidate': None, 'count': 0}


def update_gesture_stability(state, raw_gesture, stable_frames):
    """
    Anti-flicker debounce for one hand's gesture (spec section 13). `state`
    is mutated in place so the caller just keeps reusing the same dict.

    A gesture only becomes "confirmed" -- replacing whatever was confirmed
    before -- after showing up for `stable_frames` consecutive frames. That
    applies in both directions: appearing AND disappearing need to be stable.
    """
    if raw_gesture == state['confirmed']:
        state['candidate'] = None
        state['count'] = 0
        return state['confirmed']

    if raw_gesture == state['candidate']:
        state['count'] += 1
    else:
        state['candidate'] = raw_gesture
        state['count'] = 1

    if state['count'] >= stable_frames:
        state['confirmed'] = state['candidate']
        state['candidate'] = None
        state['count'] = 0

    return state['confirmed']


def update_space_state(raw_middle_active, space_state, last_off_time, cooldown, now):
    """
    Pure state transition for the SPACE (nitro) key. SPACE stays held for as
    long as a MIDDLE gesture is confirmed on either hand; after a release it
    enforces a minimum cooldown before firing again, so jitter right at the
    edge of the gesture can't machine-gun Space.

    `last_off_time` is None until the first release ever happens, which
    always counts as "cooldown satisfied" so the very first activation of
    the whole session isn't blocked.

    Returns (new_space_state, new_last_off_time).
    """
    if raw_middle_active and not space_state:
        if last_off_time is None or (now - last_off_time) >= cooldown:
            return True, last_off_time
        return False, last_off_time
    if not raw_middle_active and space_state:
        return False, now
    return space_state, last_off_time


def determine_steering(smoothed_angle, previous_direction, dead_zone, hysteresis):
    """
    Converts a smoothed relative angle into a discrete 'LEFT' / 'RIGHT' /
    'CENTER' direction, with hysteresis so the output doesn't chatter when
    the angle sits right on the dead-zone boundary.
    """
    enter = dead_zone
    leave = max(dead_zone - hysteresis, 0.0)

    if previous_direction == 'RIGHT' and smoothed_angle >= leave:
        return 'RIGHT'
    if previous_direction == 'LEFT' and smoothed_angle <= -leave:
        return 'LEFT'

    if smoothed_angle >= enter:
        return 'RIGHT'
    if smoothed_angle <= -enter:
        return 'LEFT'
    return 'CENTER'


# ============================================================
# KEYBOARD CONTROLLER
# (the one deliberate class in this file -- centralizing key state is
# exactly what spec section 14 asks for, and it's genuinely simpler than
# threading raw press/release calls through every call site)
# ============================================================

class KeyboardController:
    """
    Centralized keyboard state. Every key press/release goes through here so
    we can guarantee: A and D are never held together, we never send a
    redundant press/release for a key that's already in that state, and
    release_all() always leaves the keyboard clean.
    """

    def __init__(self):
        self._kb = PynputController()
        self.state = {'a': False, 'd': False, 's': False, 'space': False}
        self._key_objs = {'a': 'a', 'd': 'd', 's': 's', 'space': Key.space}

    def _press(self, name):
        if not self.state[name]:
            self._kb.press(self._key_objs[name])
            self.state[name] = True

    def _release(self, name):
        if self.state[name]:
            self._kb.release(self._key_objs[name])
            self.state[name] = False

    def set_steering(self, direction):
        """direction: 'LEFT', 'RIGHT', or 'CENTER'. Never presses A and D together."""
        if direction == 'LEFT':
            self._release('d')
            self._press('a')
        elif direction == 'RIGHT':
            self._release('a')
            self._press('d')
        else:
            self._release('a')
            self._release('d')

    def set_drift(self, active):
        self._press('s') if active else self._release('s')

    def set_space(self, active):
        self._press('space') if active else self._release('space')

    def release_all(self):
        for name in list(self.state.keys()):
            self._release(name)


def update_keyboard(keyboard, steering_direction, drift_active, space_active):
    keyboard.set_steering(steering_direction)
    keyboard.set_drift(drift_active)
    keyboard.set_space(space_active)


def release_all_keys(keyboard):
    keyboard.release_all()


# ============================================================
# CAMERA / MODEL / MEDIAPIPE SETUP
# ============================================================

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {CAMERA_INDEX}. Check that no other "
            "app is using the camera, that you've granted camera permission, and "
            "try a different CAMERA_INDEX (0, 1, 2...) near the top of the file."
        )
    return cap


def ensure_model_downloaded():
    """Downloads MediaPipe's hand-landmark model next to this script if it's
    not already there. Only needs to happen once."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    print(f"[INFO] Hand-tracking model not found, downloading to:\n       {model_path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not automatically download the MediaPipe hand-landmark model.\n"
            f"  Reason: {exc}\n"
            f"  Please download it manually from:\n    {MODEL_URL}\n"
            f"  and save it as:\n    {model_path}"
        ) from exc
    print("[INFO] Model download complete.")
    return model_path


def initialize_hands():
    """
    Sets up a MediaPipe HandLandmarker (current Tasks API) in VIDEO running
    mode, so detect_for_video() can be called once per webcam frame and
    return a result immediately -- no callback/threading needed.
    """
    model_path = ensure_model_downloaded()
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


_last_timestamp_ms = 0


def _next_timestamp_ms():
    """Monotonically increasing millisecond timestamp for detect_for_video(),
    which requires strictly increasing timestamps between calls."""
    global _last_timestamp_ms
    ts = int(time.time() * 1000)
    if ts <= _last_timestamp_ms:
        ts = _last_timestamp_ms + 1
    _last_timestamp_ms = ts
    return ts


def detect_hands(landmarker, rgb_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, _next_timestamp_ms())


def get_hand_landmarks(result, frame_w, frame_h):
    """
    Organizes a HandLandmarkerResult into {'Left': info_or_None, 'Right': info_or_None}.
    Each info dict carries the raw landmark list plus derived pixel-space
    center/orientation, so downstream code never touches the MediaPipe result
    object directly.
    """
    hands_info = {'Left': None, 'Right': None}
    if not result or not result.hand_landmarks:
        return hands_info

    for lm_list, handedness in zip(result.hand_landmarks, result.handedness):
        if not handedness:
            continue
        label = handedness[0].category_name  # 'Left' or 'Right'
        if label not in hands_info:
            continue
        hands_info[label] = {
            'landmarks': lm_list,
            'center': get_fist_center(lm_list, frame_w, frame_h),
            'orientation': get_hand_orientation_angle(lm_list, frame_w, frame_h),
        }
    return hands_info


# ============================================================
# VISUALIZATION
# ============================================================

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)


def draw_hand_skeleton(frame, landmarks, color):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)


def draw_interface(frame, hands_info, gestures, steering_info, keyboard_state, fps):
    left_info = hands_info['Left']
    right_info = hands_info['Right']

    # ---- skeletons + fist centers (only drawn for hands detected THIS frame) ----
    for side, info, color in (('Left', left_info, LEFT_COLOR), ('Right', right_info, RIGHT_COLOR)):
        if info is None:
            continue
        draw_hand_skeleton(frame, info['landmarks'], color)
        cx, cy = info['center']
        cv2.circle(frame, (cx, cy), 8, color, -1)
        put_text(frame, side[0], (cx - 6, cy - 14), color, 0.5)
        put_text(frame, gestures[side].replace('_', ' '), (cx - 40, cy + 35), color, 0.5)

    # ---- steering line + horizontal reference ----
    # Uses whatever positions steering actually computed from this frame
    # (which may briefly be a cached position -- see main()), so the line
    # never flickers out just because of a single missed detection.
    ll, lr = steering_info['line_left'], steering_info['line_right']
    if ll is not None and lr is not None:
        line_color = (0, 255, 0) if steering_info['mode'] == 'TWO_HAND' else (130, 130, 130)
        cv2.line(frame, ll, lr, line_color, 2, cv2.LINE_AA)
        mid = ((ll[0] + lr[0]) // 2, (ll[1] + lr[1]) // 2)
        cv2.line(frame, (mid[0] - 120, mid[1]), (mid[0] + 120, mid[1]), (90, 90, 90), 1, cv2.LINE_AA)

    # ---- text panel ----
    raw_angle = steering_info['raw_angle']
    lines = [
        f"Left Hand: {'DETECTED' if left_info else 'NOT DETECTED'}",
        f"Right Hand: {'DETECTED' if right_info else 'NOT DETECTED'}",
        f"Left Gesture: {gestures['Left'].replace('_', ' ')}",
        f"Right Gesture: {gestures['Right'].replace('_', ' ')}",
        "",
        f"Steering Angle: {raw_angle:+.1f} deg" if raw_angle is not None else "Steering Angle: --",
        f"Center Angle: {steering_info['center_angle']:+.1f} deg",
        f"Relative Angle: {steering_info['relative_angle']:+.1f} deg",
        f"Direction: {steering_info['direction']}",
        f"Mode: {steering_info['mode']}",
        "",
        f"A: {'ON' if keyboard_state['a'] else 'OFF'}",
        f"D: {'ON' if keyboard_state['d'] else 'OFF'}",
        f"S: {'ON' if keyboard_state['s'] else 'OFF'}",
        f"SPACE: {'ON' if keyboard_state['space'] else 'OFF'}",
        "",
        f"FPS: {fps:.0f}",
    ]

    y = 22
    for line in lines:
        if not line:
            y += 10
            continue
        color = (255, 255, 255)
        if line.endswith(': ON'):
            color = (0, 255, 0)
        elif line.endswith(': OFF'):
            color = (170, 170, 170)
        elif line.endswith('NOT DETECTED'):
            color = (0, 0, 255)
        elif line.endswith('DETECTED'):
            color = (0, 255, 0)
        put_text(frame, line, (10, y), color, 0.5)
        y += 20

    put_text(frame, "Q: Quit   R: Recalibrate", (10, frame.shape[0] - 12), (0, 255, 255), 0.5)


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_center(cap, landmarker, fallback=(0.0, 0.0, 0.0)):
    """
    Shows a countdown, then samples the two-fist angle and each hand's own
    orientation while the user holds both fists in a normal straight-driving
    pose. Returns (center_angle, left_ref_angle, right_ref_angle) in degrees.

    If the user presses 'q' before any usable sample is gathered, returns
    `fallback` instead -- pass in the *current* values when recalibrating so
    a canceled recalibration doesn't reset an already-good setup to zero.
    """
    # --- countdown ---
    countdown_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        remaining = CALIBRATION_COUNTDOWN - (time.time() - countdown_start)
        if remaining <= 0:
            break
        put_text(frame, "CALIBRATION", (20, 40), (0, 255, 255), 0.9, 2)
        put_text(frame, "Hold BOTH hands as FISTS in your normal", (20, 80), (255, 255, 255), 0.6)
        put_text(frame, "straight-driving position.", (20, 105), (255, 255, 255), 0.6)
        put_text(frame, f"Starting in {remaining:0.1f}s...", (20, 150), (0, 255, 0), 0.8)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Calibration skipped.")
            return fallback

    # --- sampling ---
    two_hand_samples, left_samples, right_samples = [], [], []
    sample_start = time.time()
    while time.time() - sample_start < CALIBRATION_SAMPLE_TIME:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detect_hands(landmarker, rgb)
        hands_info = get_hand_landmarks(result, w, h)

        left_info, right_info = hands_info['Left'], hands_info['Right']
        if left_info is not None:
            draw_hand_skeleton(frame, left_info['landmarks'], LEFT_COLOR)
        if right_info is not None:
            draw_hand_skeleton(frame, right_info['landmarks'], RIGHT_COLOR)

        if left_info is not None and right_info is not None:
            if detect_fist(get_finger_states(left_info['landmarks'])) and \
               detect_fist(get_finger_states(right_info['landmarks'])):
                two_hand_samples.append(calculate_steering_angle(left_info['center'], right_info['center']))
                left_samples.append(left_info['orientation'])
                right_samples.append(right_info['orientation'])

        remaining = CALIBRATION_SAMPLE_TIME - (time.time() - sample_start)
        put_text(frame, "CALIBRATING... hold still", (20, 40), (0, 255, 255), 0.8)
        put_text(frame, f"Time left: {remaining:0.1f}s   Samples: {len(two_hand_samples)}", (20, 70), (255, 255, 255), 0.6)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Calibration stopped early, using samples gathered so far.")
            break

    if two_hand_samples:
        center_angle = sum(two_hand_samples) / len(two_hand_samples)
        left_ref = sum(left_samples) / len(left_samples)
        right_ref = sum(right_samples) / len(right_samples)
        print(f"[INFO] Calibration done -> center={center_angle:.2f} deg, "
              f"left_ref={left_ref:.2f} deg, right_ref={right_ref:.2f} deg")
        return center_angle, left_ref, right_ref

    print("[WARN] No two-fist pose was seen during calibration; keeping previous/default values.")
    return fallback


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("[INFO] Starting Asphalt 8 hand controller...")
    landmarker = None
    cap = None
    keyboard = None
    try:
        print("[INFO] Loading MediaPipe HandLandmarker (first run may download the model)...")
        landmarker = initialize_hands()
        cap = initialize_camera()
        keyboard = KeyboardController()

        left_stability = new_gesture_stability_state()
        right_stability = new_gesture_stability_state()
        last_known = {'Left': None, 'Right': None}  # smooths over single-frame tracking misses

        steering_direction = 'CENTER'
        smoothed_angle = 0.0
        last_mode = 'NONE'
        space_state = False
        last_space_off_time = None
        frame_times = deque(maxlen=20)

        center_angle, left_ref_angle, right_ref_angle = calibrate_center(cap, landmarker)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read a frame from the camera; stopping.")
                break

            frame = cv2.flip(frame, 1)  # mirror: moving your hand right moves it right on screen
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = detect_hands(landmarker, rgb)
            hands_info = get_hand_landmarks(result, w, h)

            for side in ('Left', 'Right'):
                if hands_info[side] is not None:
                    last_known[side] = hands_info[side]

            # ---- per-hand gesture classification + debounce ----
            raw_gestures = {}
            for side in ('Left', 'Right'):
                info = hands_info[side]
                raw_gestures[side] = 'NOT DETECTED' if info is None else classify_gesture(get_finger_states(info['landmarks']))

            left_gesture = update_gesture_stability(left_stability, raw_gestures['Left'], GESTURE_STABLE_FRAMES)
            right_gesture = update_gesture_stability(right_stability, raw_gestures['Right'], GESTURE_STABLE_FRAMES)

            # Trust cached position data only while the debouncer still
            # considers that hand present -- keeps steering smooth through a
            # one-frame miss without ever acting on truly stale data.
            left_eff = last_known['Left'] if left_gesture != 'NOT DETECTED' else None
            right_eff = last_known['Right'] if right_gesture != 'NOT DETECTED' else None

            # ---- independent gestures: drift (S) and nitro (SPACE) ----
            drift_active = (left_gesture == 'OPEN_PALM') or (right_gesture == 'OPEN_PALM')
            middle_raw = (left_gesture == 'MIDDLE') or (right_gesture == 'MIDDLE')
            now = time.time()
            space_state, last_space_off_time = update_space_state(
                middle_raw, space_state, last_space_off_time, SPACE_COOLDOWN, now
            )

            # ---- steering: pick a mode from the hand-role logic (spec sections 10/11) ----
            mode = 'NONE'
            raw_angle = None
            reference_angle = 0.0
            line_left = line_right = None

            # Note: the fallback modes require the OTHER hand to be confirmed
            # present with some gesture (OPEN_PALM/MIDDLE/UNKNOWN) -- not
            # merely "not FIST", which would also match 'NOT DETECTED' and
            # incorrectly fall back to single-hand steering when a hand has
            # simply left the frame. Per section 6, a truly missing hand
            # must centre the steering, not trigger the fallback.
            NOT_FIST_BUT_PRESENT = ('OPEN_PALM', 'MIDDLE', 'UNKNOWN')
            if left_gesture == 'FIST' and right_gesture == 'FIST' and left_eff and right_eff:
                mode = 'TWO_HAND'
                raw_angle = calculate_steering_angle(left_eff['center'], right_eff['center'])
                reference_angle = center_angle
                line_left, line_right = left_eff['center'], right_eff['center']
            elif left_gesture == 'FIST' and right_gesture in NOT_FIST_BUT_PRESENT and left_eff:
                mode = 'LEFT_ONLY'
                raw_angle = left_eff['orientation']
                reference_angle = left_ref_angle
                if right_eff:
                    line_left, line_right = left_eff['center'], right_eff['center']
            elif right_gesture == 'FIST' and left_gesture in NOT_FIST_BUT_PRESENT and right_eff:
                mode = 'RIGHT_ONLY'
                raw_angle = right_eff['orientation']
                reference_angle = right_ref_angle
                if left_eff:
                    line_left, line_right = left_eff['center'], right_eff['center']

            if mode != 'NONE':
                relative_angle = normalize_angle(raw_angle - reference_angle) * STEERING_SENSITIVITY
                if INVERT_STEERING:
                    relative_angle = -relative_angle
                # Resync instantly on a mode switch instead of blending across
                # it -- the two-hand angle and single-hand orientation aren't
                # the same signal, so smoothing between them would just cause
                # a wobble.
                smoothed_angle = relative_angle if mode != last_mode else smooth_angle(smoothed_angle, relative_angle, ANGLE_SMOOTHING)
                steering_direction = determine_steering(smoothed_angle, steering_direction, STEERING_DEAD_ZONE, STEERING_HYSTERESIS)
            else:
                smoothed_angle = 0.0
                steering_direction = 'CENTER'
            last_mode = mode

            update_keyboard(keyboard, steering_direction, drift_active, space_state)

            # ---- FPS ----
            frame_times.append(time.time())
            if len(frame_times) >= 2:
                span = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / span if span > 0 else 0.0
            else:
                fps = 0.0

            # ---- draw ----
            steering_info = {
                'mode': mode,
                'raw_angle': raw_angle,
                'center_angle': center_angle,
                'relative_angle': smoothed_angle,
                'direction': steering_direction,
                'line_left': line_left,
                'line_right': line_right,
            }
            draw_interface(frame, hands_info, {'Left': left_gesture, 'Right': right_gesture},
                            steering_info, keyboard.state, fps)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested.")
                break
            elif key == ord('r'):
                print("[INFO] Recalibrating...")
                release_all_keys(keyboard)
                steering_direction = 'CENTER'
                smoothed_angle = 0.0
                last_mode = 'NONE'
                space_state = False
                center_angle, left_ref_angle, right_ref_angle = calibrate_center(
                    cap, landmarker, fallback=(center_angle, left_ref_angle, right_ref_angle)
                )

    finally:
        if keyboard is not None:
            release_all_keys(keyboard)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        print("[INFO] Controller stopped, all keys released.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")