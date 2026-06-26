"""
dashboard/app.py
AeroGuard — Streamlit MRO Intelligence Dashboard.

Run from the project root with:
    streamlit run dashboard/app.py
"""

import sys
import os
import random
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.sensor_stream import generate_sensor_stream
from src.fusion import compute_health_index
from src.severity import estimate_severity
from src.rul_model import predict_rul
from src.recommender import get_fleet_recommendations

st.set_page_config(page_title="AeroGuard", page_icon="✈️", layout="wide")

# ---------------- SESSION STATE (keeps data persistent across reruns) ----------------
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

COMPONENTS = ["APU", "Engine", "LandingGear", "Hydraulics", "FuselageSkin"]
DEFECT_TYPES = ["crack", "corrosion", "dent", "delamination"]

st.markdown("# ✈️ AeroGuard — MRO Intelligence Dashboard")
st.markdown("**Edge AI · Predictive Maintenance · Intelligent Inspection**")
st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Scan Controls")
    aircraft_id = st.text_input("Aircraft ID", "AI-9B-737")
    component = st.selectbox("Component to Inspect", COMPONENTS)
    auto_refresh = st.checkbox("Simulate live monitoring (auto-refresh every 5s)", value=False)
    run_scan = st.button("🔍 Run Inspection Scan", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"Total scans run this session: {st.session_state.scan_count}")
    if st.session_state.scan_count > 0:
        if st.button("Reset Session Data"):
            st.session_state.scan_history = []
            st.session_state.scan_count = 0
            st.rerun()


def run_dynamic_scan(component):
    """
    Generates a fresh, slightly randomized scan result every time —
    simulates a real edge-AI inference instead of static numbers.
    """
    sensor_df = generate_sensor_stream(component, n_steps=50, seed=None)

    # Randomize which defect (if any) is "detected" this scan
    defect_detected = random.random() < 0.75  # 75% chance a defect shows up
    if defect_detected:
        defect_class = random.choice(DEFECT_TYPES)
        confidence = round(random.uniform(0.72, 0.98), 3)
        area_cm2 = round(random.uniform(1.5, 11.0), 2)
        severity_result = estimate_severity(defect_class, confidence, area_cm2)
    else:
        defect_class, confidence, area_cm2 = None, 0.0, 0.0
        severity_result = {"severity_score": round(random.uniform(0.02, 0.12), 3),
                            "severity_label": "Minor"}

    rul_days = predict_rul(component)
    # add small random walk so RUL isn't static across scans
    rul_days = max(1, rul_days + random.randint(-2, 2))

    health = compute_health_index(severity_result["severity_score"], rul_days)

    return {
        "timestamp": time.strftime("%H:%M:%S"),
        "component": component,
        "defect_class": defect_class,
        "confidence": confidence,
        "area_cm2": area_cm2,
        "severity_score": severity_result["severity_score"],
        "severity_label": severity_result["severity_label"],
        "rul_days": rul_days,
        "health_index": health["health_index"],
        "recommendation": health["recommendation"],
        "sensor_df": sensor_df,
    }


# ---------------- TRIGGER SCAN ----------------
if run_scan or auto_refresh:
    result = run_dynamic_scan(component)
    st.session_state.scan_history.append(result)
    st.session_state.scan_count += 1

# ---------------- DISPLAY ----------------
if st.session_state.scan_history:
    latest = st.session_state.scan_history[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Health Score", f"{latest['health_index']}%",
                delta=f"{latest['health_index'] - (st.session_state.scan_history[-2]['health_index'] if len(st.session_state.scan_history) > 1 else latest['health_index'])}")
    col2.metric("RUL", f"{latest['rul_days']} days")
    col3.metric("Defect Severity", f"{round(latest['severity_score']*100)}%",
                help=f"Type: {latest['defect_class'] or 'None detected'}")
    col4.metric("Action", latest["recommendation"])

    st.divider()

    if latest["defect_class"]:
        st.info(f"🔎 **Defect found:** {latest['defect_class'].title()} · "
                f"Confidence {round(latest['confidence']*100)}% · "
                f"Area {latest['area_cm2']} cm² · "
                f"Severity: **{latest['severity_label']}**")
    else:
        st.success("✅ No visible defects detected in this scan.")

    # Live sensor chart
    sensor_df = latest["sensor_df"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sensor_df["vibration"], name="Vibration", mode="lines"))
    fig.add_trace(go.Scatter(y=sensor_df["temperature"], name="Temperature",
                              mode="lines", yaxis="y2"))
    fig.update_layout(
        title=f"Sensor Telemetry — {component} (Scan #{st.session_state.scan_count})",
        height=350,
        yaxis=dict(title="Vibration (mm/s)"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if latest["recommendation"] == "GROUND IMMEDIATELY":
        st.error(f"🔴 {latest['recommendation']} — {component} requires urgent attention")
    elif latest["recommendation"] == "SCHEDULE MAINTENANCE":
        st.warning(f"🟡 {latest['recommendation']} — plan service for {component}")
    else:
        st.success(f"🟢 {latest['recommendation']} — {component} is healthy")

    st.divider()
    st.subheader("📊 Fleet-wide Maintenance Recommendations")

    # Build live recommendations from real scan history per component,
    # falling back to a fresh dynamic scan for components not yet scanned
    seen_components = {h["component"] for h in st.session_state.scan_history}
    fleet_data = []
    for c in COMPONENTS:
        history_for_c = [h for h in st.session_state.scan_history if h["component"] == c]
        if history_for_c:
            latest_for_c = history_for_c[-1]
            fleet_data.append({
                "component": c,
                "defect_severity": latest_for_c["severity_score"],
                "rul_days": latest_for_c["rul_days"],
            })
        else:
            fleet_data.append({
                "component": c,
                "defect_severity": round(random.uniform(0.05, 0.3), 2),
                "rul_days": predict_rul(c),
            })

    recs = get_fleet_recommendations(fleet_data)
    for r in recs:
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[r["priority"]]
        st.write(f"{icon} **{r['component']}** — {r['recommendation']} "
                 f"(Health: {r['health_index']}% · RUL: {r['rul_days']}d)")

    st.divider()
    st.subheader("📜 Scan History (this session)")
    history_df = pd.DataFrame([
        {k: v for k, v in h.items() if k != "sensor_df"}
        for h in st.session_state.scan_history
    ])
    st.dataframe(history_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Configure the aircraft and component in the sidebar, then click **Run Inspection Scan**.")

# Auto-refresh loop
if auto_refresh:
    time.sleep(5)
    st.rerun()