"""
src/fusion.py
Combines visual defect severity scores with sensor-based RUL predictions
into a single composite health index and maintenance recommendation.
"""


def compute_health_index(defect_severity, rul_days, max_rul=120):
    """
    Fuse visual defect severity with sensor RUL into a composite score.

    Args:
        defect_severity (float): 0 (no defect) to 1 (critical defect)
        rul_days (int/float): predicted remaining useful life in days
        max_rul (int): the RUL value considered "fully healthy" (normalizer)

    Returns:
        dict with health_index (0-100), recommendation, rul_days, defect_severity
    """
    rul_score = min(rul_days / max_rul, 1.0)
    defect_score = 1 - defect_severity

    # Weighted fusion — defects weighted slightly higher than RUL alone,
    # since a visible structural defect is often more urgent than a
    # gradual sensor trend.
    health_index = round((0.45 * rul_score + 0.55 * defect_score) * 100)

    if health_index < 30 or rul_days < 10:
        recommendation = "GROUND IMMEDIATELY"
    elif health_index < 60 or rul_days < 30:
        recommendation = "SCHEDULE MAINTENANCE"
    else:
        recommendation = "MONITOR"

    return {
        "health_index": health_index,
        "recommendation": recommendation,
        "rul_days": rul_days,
        "defect_severity": defect_severity,
    }


if __name__ == "__main__":
    # Quick test: python src/fusion.py
    result = compute_health_index(defect_severity=0.82, rul_days=18)
    print(result)
