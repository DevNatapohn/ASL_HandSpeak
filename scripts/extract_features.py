# ASL_Project\scripts\extract_features.py

import pandas as pd  # type: ignore
import numpy as np

IN_CSV  = "./data/asl_landmarks_xyz.csv"
OUT_CSV = "./data/asl_features.csv"

# เลือก landmark ที่สำคัญ (ข้อมือ + ปลายนิ้ว)
KEYPOINTS = [0, 4, 8, 12, 16, 20]

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def angle(v1, v2):
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def main():
    df = pd.read_csv(IN_CSV)
    feats = []
    for _, row in df.iterrows():
        coords = np.array([[row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]] for i in range(21)])
        wrist = coords[0]

        fvec = []

        # ------------------------------
        # 1) ระยะจากข้อมือ → ปลายนิ้ว
        for i in KEYPOINTS[1:]:
            fvec.append(distance(wrist, coords[i]))

        # ------------------------------
        # 2) ระยะระหว่างปลายนิ้วแต่ละคู่
        for i in KEYPOINTS[1:]:
            for j in KEYPOINTS[1:]:
                if i < j:
                    fvec.append(distance(coords[i], coords[j]))

        # ------------------------------
        # 3) มุมระหว่างนิ้ว (vectors from wrist)
        v_index  = coords[8]  - wrist
        v_middle = coords[12] - wrist
        v_ring   = coords[16] - wrist
        v_pinky  = coords[20] - wrist

        fvec.append(angle(v_index, v_middle))
        fvec.append(angle(v_index, v_ring))
        fvec.append(angle(v_index, v_pinky))
        fvec.append(angle(v_middle, v_ring))
        fvec.append(angle(v_middle, v_pinky))
        fvec.append(angle(v_ring, v_pinky))

        # ------------------------------
        # 4) Dot product (normalized)
        def norm_dot(a, b):
            return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6)

        fvec.append(norm_dot(v_index, v_middle))
        fvec.append(norm_dot(v_index, v_ring))
        fvec.append(norm_dot(v_index, v_pinky))
        fvec.append(norm_dot(v_middle, v_ring))
        fvec.append(norm_dot(v_middle, v_pinky))
        fvec.append(norm_dot(v_ring, v_pinky))

        # ------------------------------
        # 5) Cross product magnitude
        def cross_mag(a, b):
            return np.linalg.norm(np.cross(a, b))

        fvec.append(cross_mag(v_index, v_middle))
        fvec.append(cross_mag(v_index, v_ring))
        fvec.append(cross_mag(v_index, v_pinky))
        fvec.append(cross_mag(v_middle, v_ring))
        fvec.append(cross_mag(v_middle, v_pinky))
        fvec.append(cross_mag(v_ring, v_pinky))

        # ------------------------------
        # 6) Ratio ของระยะทาง (index/middle, index/ring, ...)
        d_im = distance(coords[8], coords[12]) + 1e-6
        d_ir = distance(coords[8], coords[16]) + 1e-6
        d_ip = distance(coords[8], coords[20]) + 1e-6
        fvec.append(d_im / d_ir)
        fvec.append(d_im / d_ip)
        fvec.append(d_ir / d_ip)

        # ------------------------------
        # 7) Z-difference (ระดับความลึกของนิ้วกับข้อมือ)
        for i in KEYPOINTS[1:]:
            fvec.append(coords[i][2] - wrist[2])

        # ------------------------------
        # 8) Difference of coordinates (x,y,z ระหว่างนิ้วกับข้อมือ)
        for i in KEYPOINTS[1:]:
            diff = coords[i] - wrist
            fvec.extend(diff)  # (dx, dy, dz)

        # ------------------------------
        # 9) Vector components (เก็บเวกเตอร์จากข้อมือไปหานิ้ว)
        for i in KEYPOINTS[1:]:
            vec = coords[i] - wrist
            fvec.extend(vec)  # (vx, vy, vz)

        feats.append([row["label"]] + fvec)

    # ------------------------------
    # สร้าง header
    header = ["label"] + [f"f{i}" for i in range(len(feats[0]) - 1)]
    out_df = pd.DataFrame(feats, columns=header)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Features saved → {OUT_CSV}")

if __name__ == "__main__":
    main()
