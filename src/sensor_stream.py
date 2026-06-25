"""
src/sensor_stream.py
Simulates real-time sensor readings (vibration, temperature, pressure)
for aircraft components. In a real edge deployment, this would be
replaced by actual MQTT sensor feeds.
"""

import numpy as np
import pandas as pd
import time


def generate_sensor_stream(component="APU", n_steps=100, seed=None):
    """
    Simulate a time-series of sensor readings for a given component.

    Args:
        component (str): One of "APU", "Engine", "LandingGear"
        n_steps (int): Number of time steps to simulate
        seed (int): Optional random seed for reproducibility

    Returns:
        pd.DataFrame with columns: timestamp, component, vibration,
        temperature, pressure
    """
    if seed is not None:
        np.random.seed(seed)

    base = {
        "APU": {"vibration": 3.2, "temp": 380, "pressure": 3000},
        "Engine": {"vibration": 1.8, "temp": 410, "pressure": 3100},
        "LandingGear": {"vibration": 0.9, "temp": 45, "pressure": 2900},
    }

    b = base.get(component, base["APU"])
    readings = []
    now = time.time()

    for i in range(n_steps):
        noise = np.random.normal(0, 0.05)
        degradation = i * 0.01  # simulate gradual wear over time
        readings.append({
            "timestamp": now + i,
            "component": component,
            "vibration": round(b["vibration"] * (1 + degradation + noise), 3),
            "temperature": round(b["temp"] * (1 + degradation * 0.5 + noise), 1),
            "pressure": round(b["pressure"] * (1 - degradation * 0.3 + noise), 1),
        })

    return pd.DataFrame(readings)


def stream_live_reading(component="APU"):
    """
    Returns a single simulated live sensor reading (for dashboard polling).
    """
    df = generate_sensor_stream(component, n_steps=1)
    return df.iloc[0].to_dict()


if __name__ == "__main__":
    # Quick test when running this file directly:
    # python src/sensor_stream.py
    df = generate_sensor_stream("APU", n_steps=10)
    print(df)
