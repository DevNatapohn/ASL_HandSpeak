# ✋ ASL_HandSpeak – American Sign Language Recognition

**ASL_HandSpeak** is a real-time American Sign Language (ASL) recognition system built with **MediaPipe, OpenCV, and Machine Learning**.  
It can detect and classify ASL alphabet hand signs (A–Z, except J and Z due to motion requirements).  

This project is designed for learning, prototyping, and experimenting with **Computer Vision, Hand Tracking, and Gesture Recognition**.  
Users can collect their own dataset, train models, and run live ASL recognition using a webcam.  

---

## 📂 Project Structure

```bash
ASL_HandSpeak/
│── core/                     # Core modules for processing and utilities
│   ├── draw_utils.py         # Drawing landmarks and skeleton visualization
│   ├── hand_utils.py         # Hand detection & coordinate calculation helpers
│   ├── logger.py             # Logging/exporting data into CSV
│   └── preprocess.py         # Data preprocessing before feeding into the model
│
│── model/                    # Trained models and dataset storage
│   ├── asl_model.pkl         # Pre-trained ASL model (pickle format)
│   └── dataset/              # Dataset for training/testing
│
│── scripts/                  # Main scripts
│   ├── train.py              # Train a new model
│   ├── test_ASL.py           # Run real-time ASL recognition (webcam)
│   └── collect_dataset.py    # Collect hand sign data to build dataset
│
│── requirements.txt          # Python dependencies
│── .gitignore                # Ignored files for Git
│── README.md                 # Project documentation
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
