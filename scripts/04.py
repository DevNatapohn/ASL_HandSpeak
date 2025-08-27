#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 04.py

import cv2
import mediapipe as mp # type: ignore
import numpy as np
import pickle
from collections import deque

MODEL_PATH = "./model/asl_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)["model"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

KEYPOINTS = [0, 4, 8, 12, 16, 20]


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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("ไม่พบกล้อง")

pred_buffer = deque(maxlen=5)

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

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            feat = extract_features(coords)

            probs = model.predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            pred_char = model.classes_[idx]
            conf = float(probs[idx])

            pred_buffer.append(pred_char)
            vals, counts = np.unique(pred_buffer, return_counts=True)
            pred_char = vals[np.argmax(counts)]

        cv2.putText(frame, f"Prediction: {pred_char} (conf≈{conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("ASL Inference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
