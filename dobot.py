#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ASL Inference + Dobot Pick & Place + Draw Utils + gTTS + Debounce + Countdown

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import pickle
from collections import deque
import threading
import pygame # type: ignore
from gtts import gTTS # type: ignore
import time

# ------- CONFIG -------
MODEL_PATH = "./model/asl_model.pkl"
SOUNDS_DIR = "sounds"
DELAY_SEC = 1.2  # เวลารอระหว่างการ detect ตัวเดียวกัน
DOBOT_ENABLED = True
DOBOT_PORT = "/dev/ttyUSB0"
SAFE_Z = 20.0  # ระยะสูงปลอดภัยจากฐาน

if not os.path.exists(SOUNDS_DIR):
    os.makedirs(SOUNDS_DIR)

supported_chars = [chr(c) for c in range(ord('A'), ord('Z')) if chr(c) != 'J']
sound_cache = {}

# ------- โหลดโมเดล -------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)["model"]

# ------- Mediapipe -------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
KEYPOINTS = [0, 4, 8, 12, 16, 20]

# ------- ฟังก์ชันฟีเจอร์ -------
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def angle(v1, v2):
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def extract_features(coords):
    wrist = coords[0]
    fvec = []
    for i in KEYPOINTS[1:]:
        fvec.append(distance(wrist, coords[i]))
    for i in KEYPOINTS[1:]:
        for j in KEYPOINTS[1:]:
            if i < j:
                fvec.append(distance(coords[i], coords[j]))
    v_index = coords[8] - wrist
    v_middle = coords[12] - wrist
    v_ring = coords[16] - wrist
    v_pinky = coords[20] - wrist
    fvec.append(angle(v_index, v_middle))
    fvec.append(angle(v_index, v_ring))
    fvec.append(angle(v_index, v_pinky))
    fvec.append(angle(v_middle, v_ring))
    fvec.append(angle(v_middle, v_pinky))
    fvec.append(angle(v_ring, v_pinky))
    return np.array(fvec).reshape(1, -1)

# ------- gTTS / pygame -------
pygame.init()
try:
    pygame.mixer.init()
except Exception as e:
    print("Warning: pygame.mixer.init() failed:", e)

def ensure_sound_for_char(char):
    if char in sound_cache and sound_cache[char] is not None:
        return sound_cache[char]
    filename = os.path.join(SOUNDS_DIR, f"{char}.mp3")
    if not os.path.exists(filename):
        try:
            tts = gTTS(text=char, lang="en")
            tts.save(filename)
        except Exception as e:
            print(f"gTTS failed for {char}: {e}")
            return None
    try:
        sound = pygame.mixer.Sound(filename)
        sound_cache[char] = sound
        return sound
    except Exception as e:
        print(f"Failed to load sound {filename}: {e}")
        sound_cache[char] = None
        return None

for ch in supported_chars:
    ensure_sound_for_char(ch)

# ------- 4D positions -------
positions_4d = {
    "home": (12.75, 97.28, 1.74, 82.53),
    "block_1": (48.28, 146.26, -41.27, 71.73),
    "block_2": (61.18, 193.10, -43.19, 105.56),
    "drop_off_1": (61.18, 193.10, -43.19, 72.42),
    "drop_off_2": (61.18, 193.10, -43.19, 102.12),
    "A": (200, 0, 50, 0),
    "B": (200, 100, 50, 0),
}

# ------- Dobot -------
dobot_client = None
dobot_lib_name = None
if DOBOT_ENABLED:
    try:
        from pydobot import Dobot # type: ignore
        dobot_lib_name = "pydobot"
        dobot_client = Dobot(port=DOBOT_PORT, verbose=False)
        print("Connected to Dobot")
    except Exception as e:
        print("Dobot connection failed:", e)
        dobot_client = None

def dobot_move_to(x, y, z, r):
    if not DOBOT_ENABLED or dobot_client is None:
        print(f"Move skipped to ({x},{y},{z},{r})")
        return
    try:
        dobot_client.move_to(x, y, z, r)
    except Exception as e:
        print("Error dobot move:", e)

def dobot_grip(on=True):
    if not DOBOT_ENABLED or dobot_client is None:
        print(f"Grip {'ON' if on else 'OFF'} skipped")
        return
    try:
        dobot_client.gripper(on)
    except Exception as e:
        print("Error controlling gripper:", e)

def move_to_position(pos_name, z_override=None):
    if pos_name in positions_4d:
        x, y, z, r = positions_4d[pos_name]
        if z_override is not None:
            z = z_override
        print(f"Moving to {pos_name}: x={x}, y={y}, z={z}, r={r}")
        dobot_move_to(x, y, z, r)
    else:
        print(f"Position '{pos_name}' not found")

# ------- Mapping A–P / Q–Y -------
def get_position_name_from_char(char):
    if char in [chr(c) for c in range(ord('A'), ord('P')+1)]:
        return ("block_1", "drop_off_1")
    elif char in [chr(c) for c in range(ord('Q'), ord('Y')+1)]:
        return ("block_2", "drop_off_2")
    else:
        return (None, None)

# ------- Draw Utils -------
def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),(17,0)
        ]
        for start,end in connections:
            cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0,0,0),6)
            cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255,255,255),2)
    for idx, lm in enumerate(landmark_point):
        radius = 8 if idx in [4,8,12,16,20] else 5
        cv2.circle(image, (lm[0], lm[1]), radius, (255,255,255), -1)
        cv2.circle(image, (lm[0], lm[1]), radius, (0,0,0),1)
    return image

def draw_bounding_rect(image, brect):
    cv2.rectangle(image, (brect[0],brect[1]), (brect[2],brect[3]), (0,0,0),1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, conf):
    cv2.rectangle(image, (brect[0], brect[1]-30), (brect[2], brect[1]), (0,0,0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text != "":
        info_text += f":{hand_sign_text} ({conf:.0f}%)"
    cv2.putText(image, info_text, (brect[0]+5,brect[1]-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1,cv2.LINE_AA)
    return image

def draw_fps(image, fps):
    cv2.putText(image, f"FPS:{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(image, f"FPS:{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)

# ------- Camera + Main Loop -------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found")

pred_buffer = deque(maxlen=5)
last_pred_char = ""
last_detect_time = 0
prev_time = 0
current_pos_name = None

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    print("Starting camera. Show ASL letter...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pred_char = "-"
        conf = 0.0
        h, w, _ = frame.shape
        landmark_point = []

        current_time = time.time()
        remaining_delay = max(0, DELAY_SEC - (current_time - last_detect_time))

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                landmark_point.append([int(lm.x*w), int(lm.y*h)])

            frame = draw_landmarks(frame, landmark_point)
            brect = [min([p[0] for p in landmark_point]), min([p[1] for p in landmark_point]),
                     max([p[0] for p in landmark_point]), max([p[1] for p in landmark_point])]
            frame = draw_bounding_rect(frame, brect)

            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            feat = extract_features(coords)

            try:
                probs = model.predict_proba(feat)[0]
                idx = int(np.argmax(probs))
                pred_char = model.classes_[idx]
                conf = float(probs[idx])*100
            except Exception as e:
                pred_char = "-"
                conf = 0.0

            pred_buffer.append(pred_char)
            vals, counts = np.unique(pred_buffer, return_counts=True)
            pred_char = vals[np.argmax(counts)]

            # Debounce / delay + Detected Log
            if remaining_delay <= 0 and pred_char in supported_chars and pred_char != last_pred_char:
                print(f"\nDetected: {pred_char} (conf≈{conf:.2f})")
                last_pred_char = pred_char
                last_detect_time = current_time
                sound_obj = ensure_sound_for_char(pred_char)
                if sound_obj:
                    threading.Thread(target=lambda s: s.play(), args=(sound_obj,), daemon=True).start()

                # Dobot Pick & Place
                pos_start, pos_end = get_position_name_from_char(pred_char)
                if pos_start and pos_end:
                    move_to_position(pos_start, z_override=SAFE_Z)
                    time.sleep(1)
                    move_to_position(pos_start)
                    time.sleep(1)
                    dobot_grip(True)
                    time.sleep(0.5)
                    move_to_position(pos_start, z_override=SAFE_Z)
                    time.sleep(1)
                    move_to_position(pos_end, z_override=SAFE_Z)
                    time.sleep(1)
                    move_to_position(pos_end)
                    time.sleep(1)
                    dobot_grip(False)
                    time.sleep(0.5)
                    move_to_position(pos_end, z_override=SAFE_Z)
                    current_pos_name = pos_end

            frame = draw_info_text(frame, brect, results.multi_handedness[0], pred_char, conf)

        # วัด FPS
        curr_time = time.time()
        fps = 1/(curr_time - prev_time + 1e-6)
        prev_time = curr_time
        draw_fps(frame, fps)

        # แสดง countdown ข้างๆ FPS
        if remaining_delay > 0:
            cv2.putText(frame, f"Wait: {remaining_delay:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # Display Prediction & Position
        cv2.putText(frame, f"Prediction: {pred_char} (conf≈{conf:.2f})", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        if current_pos_name:
            cv2.putText(frame, f"Position: {current_pos_name}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

        cv2.imshow("ASL Dobot Safe Pick & Place", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
if dobot_client is not None:
    try:
        dobot_client.disconnect()
    except Exception:
        pass
