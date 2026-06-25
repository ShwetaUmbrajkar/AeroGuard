"""
dashboard/app.py
AeroGuard — Streamlit MRO Intelligence Dashboard.

Run from the project root with:
    streamlit run dashboard/app.py
"""

import sys
import os

# --- IMPORT FIX ---
# Streamlit only adds the folder containing this script (dashboard/) to
# sys.path, not the project root. This line adds the project root so
# "from src.xxx import yyy" works regardless of where you launch from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go

from src.sensor_stream import generate_sensor_stream
from src.fusion import compute_health_index
from src.severity import estimate_severity
from src.rul_model import predict_rul
from src.recommender import get_fleet_recommendations


st.set_page_config(page_title="AeroGuard", page_icon="✈️", layout="wide")

st.markdown("# ✈️ AeroGuard — MRO Intelligence Dashboard")
st.markdown("**Edge AI · Predictive Maintenance · Intelligent Inspection**")
st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Scan Controls")
    aircraft_id = st.text_input("Aircraft ID", "AI-9B-737")
    component = st.selectbox(
        "Component", ["APU", "Engine", "LandingGear", "Hydraulics", "FuselageSkin"]
    )
    run_scan = st.button("Run Inspection Scan", type="primary")

# ---------------- MAIN AREA ----------------
col1, col2, col3, col4 = st.columns(4)

if run_scan:
    # 1. Simulate sensor stream for the selected component
    sensor_df = generate_sensor_stream(component, n_steps=50)

    # 2. Simulated defect detection output (replace with real detect_defects() once trained)
    severity_result = estimate_severity("crack", confidence=0.96, area_cm2=4.2)
    defect_severity = severity_result["severity_score"]

    # 3. Predict RUL (mock until LSTM trained — see src/rul_model.py)
    rul_days = predict_rul(component)

    # 4. Fuse both signals into one health index
    result = compute_health_index(defect_severity, rul_days)

    col1.metric("Health Score", f"{result['health_index']}%")
    col2.metric("RUL", f"{result['rul_days']} days")
    col3.metric("Defect Severity", f"{round(defect_severity * 100)}%")
    col4.metric("Action", result["recommendation"])

    st.divider()

    # Sensor trend chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sensor_df["vibration"], name="Vibration", mode="lines"))
    fig.add_trace(go.Scatter(y=sensor_df["temperature"], name="Temperature", mode="lines", yaxis="y2"))
    fig.update_layout(
        title=f"Sensor Telemetry — {component}",
        height=350,
        yaxis=dict(title="Vibration (mm/s)"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if result["recommendation"] == "GROUND IMMEDIATELY":
        st.error(f"🔴 {result['recommendation']} — {component} requires urgent attention")
    elif result["recommendation"] == "SCHEDULE MAINTENANCE":
        st.warning(f"🟡 {result['recommendation']} — plan service for {component}")
    else:
        st.success(f"🟢 {result['recommendation']} — {component} is healthy")

    st.divider()
    st.subheader("Fleet-wide Maintenance Recommendations")

    sample_components = [
        {"component": "APU", "defect_severity": 0.82, "rul_days": predict_rul("APU")},
        {"component": "LandingGear", "defect_severity": 0.55, "rul_days": predict_rul("LandingGear")},
        {"component": "Hydraulics", "defect_severity": 0.10, "rul_days": predict_rul("Hydraulics")},
        {"component": "Engine", "defect_severity": 0.05, "rul_days": predict_rul("Engine")},
    ]
    recs = get_fleet_recommendations(sample_components)

    for r in recs:
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[r["priority"]]
        st.write(
            f"{icon} **{r['component']}** — {r['recommendation']} "
            f"(Health: {r['health_index']}% · RUL: {r['rul_days']}d)"
        )
else:
    st.info("👈 Configure the aircraft and component in the sidebar, then click **Run Inspection Scan**.")
