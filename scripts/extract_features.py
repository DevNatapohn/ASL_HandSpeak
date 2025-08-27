#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 02.py

import pandas as pd # type: ignore
import numpy as np

IN_CSV  = "./data/asl_landmarks_xyz.csv"
OUT_CSV = "./data/asl_features.csv"

# เลือก landmark ที่สำคัญ (ข้อมือ + ปลายนิ้ว)
KEYPOINTS = [0, 4, 8, 12, 16, 20]

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def angle(v1, v2):
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def main():
    df = pd.read_csv(IN_CSV)
    feats = []
    for _, row in df.iterrows():
        coords = np.array([[row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]] for i in range(21)])
        wrist = coords[0]

        fvec = []

        # 1) ระยะจากข้อมือ → ปลายนิ้ว
        for i in KEYPOINTS[1:]:
            fvec.append(distance(wrist, coords[i]))

        # 2) ระยะระหว่างปลายนิ้วแต่ละคู่
        for i in KEYPOINTS[1:]:
            for j in KEYPOINTS[1:]:
                if i < j:
                    fvec.append(distance(coords[i], coords[j]))

        # 3) มุมระหว่างนิ้ว (เช่น ข้อมือ→นิ้วชี้, ข้อมือ→นิ้วก้อย)
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

        feats.append([row["label"]] + fvec)

    # สร้าง header
    header = ["label"] + [f"f{i}" for i in range(len(feats[0])-1)]
    out_df = pd.DataFrame(feats, columns=header)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Features saved → {OUT_CSV}")

if __name__ == "__main__":
    main()
