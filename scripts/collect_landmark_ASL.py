#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 01-.py

import os
import cv2
import mediapipe as mp # type: ignore
import csv
import time
import numpy as np

# ======== ตั้งค่า ========
OUT_CSV = "./data/asl_landmarks_xyz.csv"
SAVE_COOLDOWN_SEC = 0.25  # กันกดรัวเกินไป

# A–I, K–Y static เท่านั้น (ตัด J,Z ออก)
CLASSES = [
    "A","B","C","D","E","F","G","H","I",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y"
]

DEBUG_KEYS = True

# ======== ฟังก์ชันช่วย ========
def ensure_csv(path):
    """สร้างไฟล์ CSV ถ้ายังไม่มี"""
    if not os.path.exists(path):
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def draw_info(img, text, y=30, color=(0,255,0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

# ======== Main ========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def main():
    ensure_csv(OUT_CSV)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("ไม่พบกล้องเว็บแคม")

    win_name = "Collect ASL Landmarks (XYZ)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    print("เริ่มเก็บข้อมูล (A–Y, ไม่มี J,Z):")
    print(" - กดปุ่มตัวอักษร A–I, K–Y เพื่อบันทึก static gesture")
    print(" - กด ESC เพื่อออก")
    print(f"ไฟล์จะถูกบันทึกที่: {os.path.abspath(OUT_CSV)}")

    last_save_time = 0.0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # เก็บ landmark (XYZ)
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            else:
                pts = None

            draw_info(frame, "พิมพ์ตัวอักษร (A–I,K–Y) เพื่อบันทึก", 30)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(10)

            if key != -1 and DEBUG_KEYS:
                print("Key pressed:", key)

            if key == 27:  # ESC
                break

            # กดตัวอักษร
            if (65 <= key <= 90) or (97 <= key <= 122):
                ch = chr(key).upper()
                if ch in CLASSES:
                    if pts is not None:
                        now = time.time()
                        if now - last_save_time >= SAVE_COOLDOWN_SEC:
                            row = [ch] + pts.flatten().tolist()
                            with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow(row)
                            last_save_time = now
                            print(f"Saved: {ch}")
                    else:
                        print(f"กด {ch} แต่ไม่เห็นมือในเฟรมนี้")

    cap.release()
    cv2.destroyAllWindows()
    print(f"ปิดโปรแกรมแล้ว บันทึกข้อมูลไว้ที่: {os.path.abspath(OUT_CSV)}")

if __name__ == "__main__":
    main()
