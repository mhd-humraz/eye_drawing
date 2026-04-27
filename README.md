 

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Tracking-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 👁️ Eye Canvas Pro

A real-time eye-controlled drawing application powered by **MediaPipe Iris Tracking**, **OpenCV**, and **Streamlit**.

Draw on screen using your **eye gaze**, and control drawing using **blinks** — no mouse, no touch.

---

## 🚀 Features

- 🎯 **Eye Gaze Tracking** — Move cursor using your eyes  
- 👁️ **Blink Detection** — Blink to toggle drawing mode  
- 🎨 **Multiple Colors & Brush Sizes**  
- 🧠 **9-Point Calibration System** for accuracy  
- ⚡ **Real-Time Performance (FPS optimized)**  
- 🧩 **Kalman Filter + Smoothing** for stable cursor  
- 🖥️ **Modern UI (Streamlit Dashboard)**  
- 🧼 **Clean Minimal Design (Updated UI)**  

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Computer Vision**: OpenCV  
- **Face & Iris Tracking**: MediaPipe  
- **Math & Processing**: NumPy  

---

## 📂 Project Structure
```
eye-canvas/
│
├── app.py # Main application
├── requirements.txt # Dependencies
├── Dockerfile # (Optional) Container setup
└── README.md # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/mhd-humraz/eye_drawing.git
cd eye_drawing
```

## ⚙️ Setup & Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
```
## ⚙️ Setup

### 2. Activate Environment

**Windows**
```bash
venv\Scripts\activate
```

### Linux / Mac
```
source venv/bin/activate
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### Run the Project
```
streamlit run app.py
```

## 🎮 How to Use

- Enable camera in sidebar  
- Sit **60–70 cm** from screen  
- Ensure **good lighting**  
- Complete calibration  
- Move eyes → cursor moves  
- 👁️ Blink → start drawing  
- 👁️ Blink again → stop drawing  

---

## 🎹 Controls

| Key   | Function        |
|-------|----------------|
| 1–7   | Change colors  |
| E     | Eraser         |
| C     | Clear canvas   |
| Space | Recalibrate    |

---

## 🧠 How It Works

### 👁️ Eye Tracking
- Uses **MediaPipe Face Mesh**  
- Detects iris landmarks  
- Calculates eye position relative to face  

### 📍 Calibration
- 9-point mapping system  
- Converts eye movement → screen coordinates  

### ✨ Smoothing
- Kalman filter for prediction  
- Median smoothing to reduce jitter  

### 👀 Blink Detection
- Uses Eye Aspect Ratio (EAR)  
- Detects blink when threshold drops  

---

## ⚠️ Requirements

- Webcam (required)  
- Good lighting  
- Stable head position during calibration  
