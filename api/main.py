"""
api/main.py
AeroGuard — FastAPI backend.

Run from the project root with:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.
"""

import sys
import os

# Same import fix as the dashboard — make project root importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from src.sensor_stream import generate_sensor_stream
from src.fusion import compute_health_index
from src.severity import estimate_severity
from src.rul_model import predict_rul
from src.recommender import get_fleet_recommendations

app = FastAPI(title="AeroGuard API", version="1.0")


class ScanRequest(BaseModel):
    aircraft_id: str
    component: str


@app.get("/")
def root():
    return {"status": "AeroGuard API is running", "docs": "/docs"}


@app.post("/scan")
def run_scan(req: ScanRequest):
    """Run a simulated inspection scan for a given aircraft + component."""
    sensor_df = generate_sensor_stream(req.component, n_steps=50)

    severity_result = estimate_severity("crack", confidence=0.96, area_cm2=4.2)
    defect_severity = severity_result["severity_score"]

    rul_days = predict_rul(req.component)
    health = compute_health_index(defect_severity, rul_days)

    return {
        "aircraft_id": req.aircraft_id,
        "component": req.component,
        "defect_severity": defect_severity,
        "severity_label": severity_result["severity_label"],
        "rul_days": rul_days,
        "health_index": health["health_index"],
        "recommendation": health["recommendation"],
        "sensor_sample": sensor_df.tail(5).to_dict(orient="records"),
    }


@app.get("/recommendations/{aircraft_id}")
def fleet_recommendations(aircraft_id: str):
    """Get prioritized maintenance recommendations across components."""
    sample_components = [
        {"component": "APU", "defect_severity": 0.82, "rul_days": predict_rul("APU")},
        {"component": "LandingGear", "defect_severity": 0.55, "rul_days": predict_rul("LandingGear")},
        {"component": "Hydraulics", "defect_severity": 0.10, "rul_days": predict_rul("Hydraulics")},
        {"component": "Engine", "defect_severity": 0.05, "rul_days": predict_rul("Engine")},
    ]
    recs = get_fleet_recommendations(sample_components)
    return {"aircraft_id": aircraft_id, "recommendations": recs}
