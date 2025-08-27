#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# dobot.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp # type: ignore
import numpy as np
import pickle
from collections import deque
import pygame # type: ignore
import time
from gtts import gTTS # type: ignore
import pydobot # type: ignore

# ------- CONFIG -------
MODEL_PATH = "./model/asl_model.pkl"
SOUNDS_DIR = "sounds"
if not os.path.exists(SOUNDS_DIR):
    os.makedirs(SOUNDS_DIR)

DOBOT_ENABLED = True
DOBOT_PORT = "/dev/ttyUSB0"
SAFE_Z = 20.0  # ระยะสูงปลอดภัยจากฐาน

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
supported_chars = [chr(c) for c in range(ord('A'), ord('Z')) if chr(c) != 'J']
sound_cache = {}

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
            print(f"gTTS saved {filename}")
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
}

# ------- Dobot -------
dobot_client = None
dobot_lib_name = None

if DOBOT_ENABLED:
    try:
        from pydobot import Dobot # type: ignore
        dobot_lib_name = "pydobot"
        try:
            dobot_client = Dobot(port=DOBOT_PORT, verbose=False)
            print("Connected to Dobot")
        except Exception as e:
            print("Dobot connection failed:", e)
            dobot_client = None
    except Exception:
        print("pydobot not available")
        dobot_client = None




def dobot_move_to(x, y, z, r):
    if not DOBOT_ENABLED or dobot_client is None:
        print(f"Move skipped to ({x},{y},{z},{r})")
        return
    try:
        if dobot_lib_name == "pydobot":
            try:
                dobot_client.move_to(x, y, z, r)
            except AttributeError:
                print("Update move_to() call to match SDK")
    except Exception as e:
        print("Error dobot move:", e)

def dobot_grip(on=True):
    if not DOBOT_ENABLED or dobot_client is None:
        print(f"Grip {'ON' if on else 'OFF'} skipped")
        return
    try:
        if dobot_lib_name == "pydobot":
            try:
                dobot_client.gripper(on)
            except AttributeError:
                print("Update gripper() call to match SDK")
    except Exception as e:
        print("Error controlling gripper:", e)

def move_to_position(pos_name, z_override=None):
    if pos_name in positions_4d:
        x, y, z, r = positions_4d[pos_name]
        if z_override is not None:
            z = z_override
        print(f"Moving to {pos_name}: x={x:.2f}, y={y:.2f}, z={z:.2f}, r={r:.2f}")
        dobot_move_to(x, y, z, r)
    else:
        print(f"Position '{pos_name}' not found")

positions_4d.update({
    "A": (200, 0, 50, 0),
    "B": (200, 100, 50, 0)
})

# ------- Mapping A–P / Q–Y -------
def get_position_name_from_char(char):
    if char in [chr(c) for c in range(ord('A'), ord('P')+1)]:
        return ("block_1", "drop_off_1")
    elif char in [chr(c) for c in range(ord('Q'), ord('Y')+1)]:
        return ("block_2", "drop_off_2")
    else:
        return (None, None)

# ------- Main loop -------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found")

current_pos_name = None
last_pred_char = ""
pred_buffer = deque(maxlen=5)

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

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            feat = extract_features(coords)

            try:
                probs = model.predict_proba(feat)[0]
                idx = int(np.argmax(probs))
                pred_char = model.classes_[idx]
                conf = float(probs[idx])
            except Exception as e:
                print("Model predict error:", e)
                pred_char = "-"
                conf = 0.0

            pred_buffer.append(pred_char)
            vals, counts = np.unique(pred_buffer, return_counts=True)
            pred_char = vals[np.argmax(counts)]

        if pred_char in supported_chars and pred_char != last_pred_char:
            print(f"\nDetected: {pred_char} (conf≈{conf:.2f})")

            # gTTS
            sound_obj = ensure_sound_for_char(pred_char)
            if sound_obj:
                try:
                    sound_obj.play()
                except Exception as e:
                    print(f"Sound error: {e}")
            # Dobot loop motion (แทน Pick & Place)
            if pred_char in supported_chars:
                print(f"Dobot loop triggered by: {pred_char}")

                for i in range(1):  # จะวน 3 รอบ (แก้เลขได้)
                    # ไปจุด A
                    move_to_position("A", z_override=SAFE_Z)
                    time.sleep(1)
                    move_to_position("A")
                    time.sleep(1)

                    # ดูด
                    dobot_grip(True)
                    time.sleep(0.5)

                    # ยกขึ้น
                    move_to_position("A", z_override=SAFE_Z)
                    time.sleep(1)

                    # ไปจุด B
                    move_to_position("B", z_override=SAFE_Z)
                    time.sleep(1)
                    move_to_position("B")
                    time.sleep(1)

                    # ปล่อย
                    dobot_grip(False)
                    time.sleep(0.5)

                    # ยกขึ้น
                    move_to_position("B", z_override=SAFE_Z)
                    time.sleep(1)

            # Dobot Pick & Place
#            pos_start, pos_end = get_position_name_from_char(pred_char)
#            if pos_start and pos_end:
#                # Move above pick
#                move_to_position(pos_start, z_override=SAFE_Z)
#                time.sleep(1)
#                # Move down pick
#                move_to_position(pos_start)
#                time.sleep(1)
#                dobot_grip(True)
#                time.sleep(0.5)
#                # Lift
#                move_to_position(pos_start, z_override=SAFE_Z)
#                time.sleep(1)
#                # Move above drop
#                move_to_position(pos_end, z_override=SAFE_Z)
#                time.sleep(1)
#                # Move down drop
#                move_to_position(pos_end)
#                time.sleep(1)
#                dobot_grip(False)
#                time.sleep(0.5)
#                # Lift
#                move_to_position(pos_end, z_override=SAFE_Z)
#                current_pos_name = pos_end
#            else:
#                print("No mapping for this character")

#            last_pred_char = pred_char

        # Display
        cv2.putText(frame, f"Prediction: {pred_char} (conf≈{conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if current_pos_name:
            cv2.putText(frame, f"Position: {current_pos_name}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

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