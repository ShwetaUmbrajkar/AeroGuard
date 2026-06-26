# ✈️ AeroGuard

**Edge AI-Powered Intelligent Inspection & Predictive Maintenance System for Aircraft MRO**

Built for **InnoVent-27** — AI at the Edge Solutions for Aerospace

---

## 📋 Overview

AeroGuard is a hybrid Edge AI system that unifies two critical MRO functions into a single decision pipeline:

1. **Intelligent Inspection** — YOLOv8-based computer vision detects surface defects (cracks, corrosion, dents, missing screws, chipped paint) on aircraft components
2. **Predictive Maintenance** — LSTM-based Remaining Useful Life (RUL) prediction from sensor telemetry (vibration, temperature, pressure)

These two signals are fused into a single **composite health index** per component, which drives automated maintenance recommendations — Monitor, Schedule, or Ground Immediately.

The entire pipeline is designed to run at the edge, without cloud dependency, making it suitable for hangar environments with limited connectivity.

---

## 🏗️ Architecture
┌─────────────────────┐

│  Aircraft Images /   │

│  Drone Camera Feed   │

└──────────┬───────────┘

│

▼

┌─────────────────────┐      ┌──────────────────────┐

│   YOLOv8 Defect      │      │   Sensor Telemetry    │

│   Detection (Edge)   │      │ (Vibration/Temp/Press)│

└──────────┬───────────┘      └──────────┬────────────┘

│                              │

▼                              ▼

┌─────────────────────┐      ┌──────────────────────┐

│  Severity Estimation │      │  LSTM RUL Prediction  │

└──────────┬───────────┘      └──────────┬────────────┘

│                              │

└──────────────┬───────────────┘

▼

┌─────────────────────┐

│   Data Fusion Layer  │

│  (Composite Health   │

│       Index)         │

└──────────┬───────────┘

▼

┌─────────────────────┐

│ Maintenance          │

│ Recommendation       │

│ Dashboard            │

└─────────────────────┘

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Defect Detection | YOLOv8 (Ultralytics) |
| Predictive Maintenance | PyTorch (LSTM) |
| Edge Deployment | ONNX Runtime |
| Backend API | FastAPI |
| Dashboard | Streamlit, Plotly |
| Data Processing | Pandas, NumPy, scikit-learn |
| Training Environment | Google Colab (T4 GPU) |
| Datasets | NASA CMAPSS (RUL), Roboflow aerospace defect dataset |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AeroGuard.git
cd AeroGuard

# 2. Create and activate virtual environment
python -m venv aeroguard_env
aeroguard_env\Scripts\activate      # Windows
source aeroguard_env/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run dashboard/app.py

# 5. (Optional) Run the backend API
uvicorn api.main:app --reload --port 8000
```

Dashboard: `http://localhost:8501`
API docs: `http://localhost:8000/docs`

---

## 📸 Demo Screenshots

*(To be added)*

---

## 👤 Team

**Shweta Umbrajkar** — Solo Developer
B.Tech Computer Science, VIIT Pune

---

## 📄 License

This project was built for the InnoVent-27 hackathon submission.