# ✋ ASL_HandSpeak – American Sign Language Recognition

**ASL_HandSpeak** is a real-time American Sign Language (ASL) recognition system built with **MediaPipe, OpenCV, and Machine Learning**.  
It can detect and classify ASL alphabet hand signs (A–Z, except J and Z due to motion requirements).  

This project is designed for learning, prototyping, and experimenting with **Computer Vision, Hand Tracking, and Gesture Recognition**.  
Users can collect their own dataset, train models, and run live ASL recognition using a webcam.  

---

## 📂 Project Structure

```bash
ASL_HandSpeak/
├── 📁 data/ – stores raw and processed datasets
│ ├── asl_landmarks_xyz.csv – raw landmarks (X, Y, Z) collected from Mediapipe
│ └── asl_features.csv – extracted mathematical features used for training
│
├── 📁 model/ – trained machine learning models
│ └── asl_model.pkl – RandomForest + Scaler model saved with pickle
│
├── 📁 scripts/ – main project scripts
│ ├── collect_landmark_ASL.py – collect hand landmarks from webcam and save to CSV
│ ├── extract_features.py – extract features (distances, angles, vectors, etc.) from landmarks
│ ├── train_asl_model.py – train RandomForest model and save as .pkl
│ └── test_ASL.py – real-time webcam inference with prediction display
│
├── 📁 sounds/ – reserved for audio feedback/alerts
│
└── requirements.txt – dependencies (OpenCV, Mediapipe, scikit-learn, etc.)
```
---

## ⚙️ Installation
# Clone repository
git clone https://github.com/DevNatapohn/ASL_HandSpeak.git
cd ASL_HandSpeak

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

---

## 🚀 Usage
1. Run real-time ASL recognition  
python scripts/test_ASL.py  

2. Collect new dataset samples  
python scripts/collect_dataset.py  

3. Train a new model  
python scripts/train.py  

---

## 🧠 Model
Uses scikit-learn classifiers trained on MediaPipe 3D hand landmarks (x, y, z)  

Supports ASL alphabets A–Z (except J and Z)  

Extendable with your own dataset and retraining  

---

## 📦 Requirements
- Python 3.8+  
- OpenCV  
- MediaPipe  
- NumPy  
- scikit-learn  
- pickle  

(Install all via `requirements.txt`)  
