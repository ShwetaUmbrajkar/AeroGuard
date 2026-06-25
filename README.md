# AeroGuard ✈️

Edge AI-powered Intelligent Inspection & Predictive Maintenance system for Aircraft MRO.

## Project Structure

```
AeroGuard/
├── data/                  # datasets (raw, processed, synthetic)
├── models/                # trained model weights
├── src/                   # core pipeline modules (importable package)
├── dashboard/             # Streamlit dashboard (app.py)
├── api/                   # FastAPI backend (main.py)
├── notebooks/             # Colab/Jupyter notebooks for training
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv aeroguard_env

# Windows
aeroguard_env\Scripts\activate
# Mac/Linux
source aeroguard_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the app

IMPORTANT: always run these commands from the **AeroGuard project root**
folder (the one containing `src/`, `dashboard/`, `api/`), not from inside
`dashboard/` or `api/`.

```bash
# Streamlit dashboard
streamlit run dashboard/app.py

# FastAPI backend (in a separate terminal)
uvicorn api.main:app --reload --port 8000
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

## Why the import error happened

Streamlit/FastAPI only know about the folder the script lives in unless
told otherwise. Both `dashboard/app.py` and `api/main.py` add the project
root to `sys.path` at the top of the file — this is what makes
`from src.sensor_stream import ...` work. The `src/__init__.py` file is
also required so Python recognizes `src` as a package.

## Training models (Google Colab)

See `notebooks/02_yolo_training.ipynb` and `notebooks/03_rul_training.ipynb`.
After training, download `best.pt` (YOLO) and `rul_lstm.pt` + `scaler.pkl`
(RUL model) and place them in:

```
models/defect_detection/best.pt
models/rul/rul_lstm.pt
models/rul/scaler.pkl
```

Until then, `src/detection.py` and `src/rul_model.py` automatically fall
back to realistic mock outputs, so the dashboard and API work end-to-end
even before training is complete.
