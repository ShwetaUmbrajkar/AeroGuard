"""
src/recommender.py
Generates human-readable maintenance recommendations from the
fused health index, for display on the dashboard / API.
"""

from src.fusion import compute_health_index


def get_recommendation(component, defect_severity, rul_days):
    """
    Build a full recommendation record for one component.

    Args:
        component (str): component name, e.g. "APU"
        defect_severity (float): 0-1 severity score from CV pipeline
        rul_days (int): predicted RUL in days

    Returns:
        dict: component, health_index, recommendation, action_label, priority
    """
    fused = compute_health_index(defect_severity, rul_days)

    action_map = {
        "GROUND IMMEDIATELY": {"action_label": "Ground", "priority": "high"},
        "SCHEDULE MAINTENANCE": {"action_label": "Schedule", "priority": "medium"},
        "MONITOR": {"action_label": "Monitor", "priority": "low"},
    }
    meta = action_map[fused["recommendation"]]

    return {
        "component": component,
        "health_index": fused["health_index"],
        "recommendation": fused["recommendation"],
        "action_label": meta["action_label"],
        "priority": meta["priority"],
        "rul_days": rul_days,
        "defect_severity": defect_severity,
    }


def get_fleet_recommendations(component_data):
    """
    Build recommendations for multiple components at once.

    Args:
        component_data (list of dict): each with keys
            "component", "defect_severity", "rul_days"

    Returns:
        list of recommendation dicts, sorted by priority (high first)
    """
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recs = [
        get_recommendation(c["component"], c["defect_severity"], c["rul_days"])
        for c in component_data
    ]
    return sorted(recs, key=lambda r: priority_order[r["priority"]])


if __name__ == "__main__":
    # Quick test: python src/recommender.py
    sample = [
        {"component": "APU", "defect_severity": 0.82, "rul_days": 7},
        {"component": "LandingGear", "defect_severity": 0.55, "rul_days": 18},
        {"component": "Hydraulics", "defect_severity": 0.1, "rul_days": 55},
    ]
    for r in get_fleet_recommendations(sample):
        print(r)
