#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ASL Inference + Full Draw Utils Style
# ตัวพัฒนาจาก 04.py

import cv2
import mediapipe as mp # type: ignore
import numpy as np
import pickle
from collections import deque
import time

MODEL_PATH = "./model/asl_model.pkl"

# โหลดโมเดล
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)["model"]

mp_hands = mp.solutions.hands

KEYPOINTS = [0, 4, 8, 12, 16, 20]

# --- ฟังก์ชันคำนวณ features ---
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

# --- ฟังก์ชันวาดสไตล์ draw_utils ---
def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # วาดเส้นนิ้วและ Palm connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),       # Thumb
            (0,5),(5,6),(6,7),(7,8),       # Index
            (0,9),(9,10),(10,11),(11,12),  # Middle
            (0,13),(13,14),(14,15),(15,16),# Ring
            (0,17),(17,18),(18,19),(19,20),# Pinky
            (5,9),(9,13),(13,17),(17,0)    # Palm
        ]
        for start,end in connections:
            cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0,0,0),6)
            cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255,255,255),2)
    # วาดจุด landmark
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

# --- Main ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("ไม่พบกล้อง")

pred_buffer = deque(maxlen=5)
prev_time = 0

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
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

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                landmark_point.append([int(lm.x*w), int(lm.y*h)])

            # วาดสไตล์ draw_utils
            frame = draw_landmarks(frame, landmark_point)
            brect = [min([p[0] for p in landmark_point]), min([p[1] for p in landmark_point]),
                     max([p[0] for p in landmark_point]), max([p[1] for p in landmark_point])]
            frame = draw_bounding_rect(frame, brect)

            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            feat = extract_features(coords)

            probs = model.predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            pred_char = model.classes_[idx]
            conf = float(probs[idx])*100  # %

            pred_buffer.append(pred_char)
            vals, counts = np.unique(pred_buffer, return_counts=True)
            pred_char = vals[np.argmax(counts)]

            # วาดข้อความ gesture + confidence
            frame = draw_info_text(frame, brect, results.multi_handedness[0], pred_char, conf)

        # วัด FPS
        curr_time = time.time()
        fps = 1/(curr_time - prev_time + 1e-6)
        prev_time = curr_time
        draw_fps(frame, fps)

        cv2.imshow("ASL Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
