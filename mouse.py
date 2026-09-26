"""Control the mouse with webcam hand gestures.

Install the dependencies with:
    python -m pip install opencv-python mediapipe pyautogui

The MediaPipe hand model is downloaded beside this file the first time the
program starts, so an internet connection is needed for that first launch.
The webcam preview is mirrored so movement feels natural. Press Q in the
preview window to exit. Move the pointer to the top-left corner to use
PyAutoGUI's emergency stop if needed.
"""

import math
import shutil
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui


# Tuning values. Increase SMOOTHING for steadier but slower cursor movement.
CAMERA_INDEX = 0
SMOOTHING = 0.22
CLICK_COOLDOWN_SECONDS = 0.65
CLICK_RELEASE_SECONDS = 0.18
SCROLL_BEND_ANGLE = 145
SCROLL_STRAIGHT_ANGLE = 155
SCROLL_TICKS_PER_DEGREE_PER_SECOND = 0.006
MAX_SCROLL_TICKS_PER_BEND = 8
MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def ensure_hand_model():
    """Download the official MediaPipe hand model once if it is not present."""
    if MODEL_PATH.is_file():
        return MODEL_PATH

    temporary_path = MODEL_PATH.with_suffix(".task.part")
    print("Downloading the MediaPipe hand model (one-time download)...")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=45) as response:
            with temporary_path.open("wb") as model_file:
                shutil.copyfileobj(response, model_file)
        if temporary_path.stat().st_size < 100_000:
            raise RuntimeError("The downloaded hand model is unexpectedly small.")
        temporary_path.replace(MODEL_PATH)
    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(
            "Could not download the MediaPipe hand model. Check your internet "
            "connection, or download hand_landmarker.task from Google's "
            "MediaPipe Hand Landmarker model page and place it beside mouse.py."
        ) from error
    print("Hand model saved to {}".format(MODEL_PATH))
    return MODEL_PATH


def point_distance(a, b):
    """Return the 2D distance between two normalized hand landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def joint_angle(a, b, c):
    """Return the angle ABC in degrees, using normalized x/y coordinates."""
    bax = a.x - b.x
    bay = a.y - b.y
    bcx = c.x - b.x
    bcy = c.y - b.y
    denominator = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if denominator == 0:
        return 0.0
    cosine = (bax * bcx + bay * bcy) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def describe_finger(angle, tip_y, pip_y, vertical_margin):
    """Classify a finger from its joint angle and fingertip direction."""
    if angle < SCROLL_BEND_ANGLE:
        return "BENT"
    if angle >= SCROLL_STRAIGHT_ANGLE:
        if tip_y < pip_y - vertical_margin:
            return "UP"
        if tip_y > pip_y + vertical_margin:
            return "DOWN"
        return "EXT"
    return "MID"


def analyze_hand(hand_landmarks, frame_width, frame_height):
    """Return finger states, joint angles, and pixel coordinates for a hand."""
    landmarks = hand_landmarks
    pixels = [
        (int(point.x * frame_width), int(point.y * frame_height))
        for point in landmarks
    ]
    vertical_margin = 0.008 * frame_height

    finger_data = {}
    for name, mcp, pip, dip, tip in (
        ("index", 5, 6, 7, 8),
        ("middle", 9, 10, 11, 12),
        ("ring", 13, 14, 15, 16),
        ("pinky", 17, 18, 19, 20),
    ):
        angle = joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        state = describe_finger(
            angle,
            pixels[tip][1],
            pixels[pip][1],
            vertical_margin,
        )
        finger_data[name] = {"angle": angle, "state": state}

    # A straight, lengthened thumb has both a relatively open IP joint and a
    # longer MCP-to-tip distance. This test does not depend on handedness or
    # on whether the webcam image has been mirrored.
    thumb_angle = joint_angle(landmarks[2], landmarks[3], landmarks[4])
    thumb_base_to_tip = point_distance(landmarks[2], landmarks[4])
    thumb_base_to_ip = point_distance(landmarks[2], landmarks[3])
    thumb_extended = (
        thumb_angle >= 150
        and thumb_base_to_tip >= 1.35 * thumb_base_to_ip
    )

    return {
        "index": finger_data["index"],
        "middle": finger_data["middle"],
        "ring": finger_data["ring"],
        "pinky": finger_data["pinky"],
        "thumb_extended": thumb_extended,
        "thumb_angle": thumb_angle,
        "index_tip": pixels[8],
    }


def draw_hand_landmarks(frame, landmarks):
    """Draw the 21 hand points and their connections on the webcam preview."""
    height, width = frame.shape[:2]
    points = [
        (
            max(0, min(width - 1, int(point.x * width))),
            max(0, min(height - 1, int(point.y * height))),
        )
        for point in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (70, 210, 120), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 2, (45, 110, 245), -1, cv2.LINE_AA)


def is_up(hand, name):
    return hand[name]["state"] == "UP"


def is_bent(hand, name):
    return hand[name]["state"] == "BENT"


def scroll_ticks(angle_before, angle_now, elapsed):
    """Scale one bend pulse by how quickly its joint angle changed."""
    if elapsed <= 0:
        return 1
    angular_speed = abs(angle_now - angle_before) / elapsed
    ticks = int(round(angular_speed * SCROLL_TICKS_PER_DEGREE_PER_SECOND))
    return max(1, min(MAX_SCROLL_TICKS_PER_BEND, ticks))


def get_gesture(hand, scroll_armed):
    """Choose one action using the requested click/scroll/move priority."""
    index_up = is_up(hand, "index")
    middle_up = is_up(hand, "middle")
    thumb_up = hand["thumb_extended"]
    middle_bent = is_bent(hand, "middle")
    ring_bent = is_bent(hand, "ring")
    pinky_bent = is_bent(hand, "pinky")
    other_fingers_bent = middle_bent and ring_bent and pinky_bent

    left_click = index_up and thumb_up and other_fingers_bent
    right_click = (
        index_up
        and is_up(hand, "pinky")
        and not thumb_up
        and middle_bent
        and ring_bent
    )
    clean_scroll_hand = not thumb_up and ring_bent and pinky_bent
    scroll_ready = index_up and middle_up and clean_scroll_hand

    if left_click:
        return "left_click", False
    if right_click:
        return "right_click", False
    if not clean_scroll_hand:
        scroll_armed = False
    if scroll_ready:
        return "scroll_ready", True
    if scroll_armed and clean_scroll_hand:
        if middle_up and is_bent(hand, "index"):
            return "scroll_down", True
        if index_up and middle_bent:
            return "scroll_up", True
        if index_up and middle_up:
            return "scroll_ready", True
        if not index_up and not middle_up:
            return "neutral", False
        return "scroll_ready", True
    if (
        index_up
        and not thumb_up
        and middle_bent
        and ring_bent
        and pinky_bent
    ):
        return "move", False
    return "neutral", False


def map_to_screen(point, frame_width, frame_height, screen_width, screen_height):
    """Map a webcam fingertip position to screen coordinates with edge room."""
    margin_x = frame_width * 0.06
    margin_y = frame_height * 0.06
    usable_x = max(1.0, frame_width - 2 * margin_x)
    usable_y = max(1.0, frame_height - 2 * margin_y)
    normalized_x = (point[0] - margin_x) / usable_x
    normalized_y = (point[1] - margin_y) / usable_y
    normalized_x = max(0.0, min(1.0, normalized_x))
    normalized_y = max(0.0, min(1.0, normalized_y))
    return (
        normalized_x * (screen_width - 1),
        normalized_y * (screen_height - 1),
    )


def draw_status(frame, action, hand, fps):
    """Draw the detected action, finger states, and quit hint."""
    action_labels = {
        "move": "Move cursor",
        "left_click": "Left click",
        "right_click": "Right click",
        "scroll_ready": "Scroll mode ready",
        "scroll_down": "Scroll down",
        "scroll_up": "Scroll up",
        "neutral": "No action",
    }
    label = action_labels.get(action, action)
    cv2.rectangle(frame, (8, 8), (430, 112), (20, 20, 20), -1)
    cv2.putText(
        frame,
        "Action: " + label,
        (20, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (80, 240, 120) if action != "neutral" else (225, 225, 225),
        2,
        cv2.LINE_AA,
    )
    if hand is not None:
        fingers = "I:{} M:{} R:{} P:{} T:{}".format(
            hand["index"]["state"],
            hand["middle"]["state"],
            hand["ring"]["state"],
            hand["pinky"]["state"],
            "EXT" if hand["thumb_extended"] else "BENT",
        )
    else:
        fingers = "No hand detected"
    cv2.putText(
        frame,
        fingers,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "FPS: {:.1f}   Q: quit".format(fps),
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )


def main():
    model_path = ensure_hand_model()
    vision_api = mp.tasks.vision
    options = vision_api.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision_api.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open webcam {}. Check that it is connected and available."
            .format(CAMERA_INDEX)
        )

    screen_width, screen_height = pyautogui.size()
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    smoothed_x = None
    smoothed_y = None
    scroll_armed = False
    click_latched = False
    click_release_started = None
    last_click_time = -CLICK_COOLDOWN_SECONDS
    previous_angles = None
    previous_frame_time = None
    previous_timestamp_ms = -1
    previous_tick_time = time.monotonic()
    fps = 0.0
    running = True

    try:
        with vision_api.HandLandmarker.create_from_options(options) as hand_landmarker:
            while running:
                success, frame = camera.read()
                if not success:
                    print("Webcam frame could not be read; stopping safely.")
                    break

                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                now = time.monotonic()
                timestamp_ms = max(int(now * 1000), previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms
                media_pipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                result = hand_landmarker.detect_for_video(
                    media_pipe_image,
                    timestamp_ms,
                )
                elapsed = (
                    now - previous_frame_time
                    if previous_frame_time is not None
                    else 0.0
                )
                previous_frame_time = now
                if now - previous_tick_time >= 0.5:
                    instant_fps = 1.0 / max(elapsed, 1e-6)
                    fps = instant_fps if fps == 0 else 0.75 * fps + 0.25 * instant_fps
                    previous_tick_time = now

                current_hand = None
                action = "neutral"
                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]
                    draw_hand_landmarks(frame, hand_landmarks)
                    current_hand = analyze_hand(
                        hand_landmarks,
                        frame_width,
                        frame_height,
                    )

                click_kind = None
                if current_hand is None:
                    scroll_armed = False
                    previous_angles = None
                else:
                    action, scroll_armed = get_gesture(current_hand, scroll_armed)
                    if action == "left_click":
                        click_kind = "left"
                    elif action == "right_click":
                        click_kind = "right"

                    if click_kind is not None:
                        click_release_started = None
                        if not click_latched:
                            if now - last_click_time >= CLICK_COOLDOWN_SECONDS:
                                try:
                                    pyautogui.click(button=click_kind)
                                    last_click_time = now
                                    click_latched = True
                                except pyautogui.FailSafeException:
                                    print("PyAutoGUI fail-safe activated; stopping.")
                                    running = False
                            else:
                                click_latched = True
                    elif click_latched:
                        if click_release_started is None:
                            click_release_started = now
                        elif now - click_release_started >= CLICK_RELEASE_SECONDS:
                            click_latched = False
                            click_release_started = None

                    if action == "move":
                        target_x, target_y = map_to_screen(
                            current_hand["index_tip"],
                            frame_width,
                            frame_height,
                            screen_width,
                            screen_height,
                        )
                        if smoothed_x is None:
                            smoothed_x, smoothed_y = target_x, target_y
                        else:
                            smoothed_x += SMOOTHING * (target_x - smoothed_x)
                            smoothed_y += SMOOTHING * (target_y - smoothed_y)
                        try:
                            pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
                        except pyautogui.FailSafeException:
                            print("PyAutoGUI fail-safe activated; stopping.")
                            running = False

                    angles = {
                        "index": current_hand["index"]["angle"],
                        "middle": current_hand["middle"]["angle"],
                    }
                    if previous_angles is not None and elapsed > 0:
                        if (
                            action == "scroll_down"
                            and is_up(current_hand, "middle")
                            and previous_angles["index"] >= SCROLL_BEND_ANGLE
                            and angles["index"] < SCROLL_BEND_ANGLE
                        ):
                            ticks = scroll_ticks(
                                previous_angles["index"],
                                angles["index"],
                                elapsed,
                            )
                            pyautogui.scroll(-ticks)
                        elif (
                            action == "scroll_up"
                            and is_up(current_hand, "index")
                            and previous_angles["middle"] >= SCROLL_BEND_ANGLE
                            and angles["middle"] < SCROLL_BEND_ANGLE
                        ):
                            ticks = scroll_ticks(
                                previous_angles["middle"],
                                angles["middle"],
                                elapsed,
                            )
                            pyautogui.scroll(ticks)
                    previous_angles = angles

                if current_hand is None and click_latched:
                    if click_release_started is None:
                        click_release_started = now
                    elif now - click_release_started >= CLICK_RELEASE_SECONDS:
                        click_latched = False
                        click_release_started = None

                draw_status(frame, action, current_hand, fps)
                cv2.imshow("Hand Mouse Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    running = False
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by keyboard interrupt.")
    except pyautogui.FailSafeException:
        print("PyAutoGUI fail-safe activated; stopping.")
