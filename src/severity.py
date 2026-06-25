"""
src/severity.py
Estimates defect severity from YOLO-style detection output
(bounding box area + class confidence) until the trained
severity classifier is plugged in.
"""


SEVERITY_WEIGHTS = {
    "crack": 0.9,
    "corrosion": 0.6,
    "dent": 0.3,
    "delamination": 0.8,
}


def estimate_severity(defect_class, confidence, area_cm2, image_area_cm2=4000):
    """
    Estimate a 0-1 severity score for a detected defect.

    Args:
        defect_class (str): defect type, e.g. "crack", "corrosion", "dent"
        confidence (float): YOLO detection confidence (0-1)
        area_cm2 (float): estimated defect area in cm^2
        image_area_cm2 (float): reference inspected surface area, for normalizing size

    Returns:
        dict with severity_score (0-1) and severity_label
    """
    class_weight = SEVERITY_WEIGHTS.get(defect_class.lower(), 0.5)
    size_ratio = min(area_cm2 / image_area_cm2, 1.0)

    # Severity blends how confident we are, how bad the defect type
    # typically is, and how large the affected area is.
    severity_score = round(
        (0.4 * confidence) + (0.4 * class_weight) + (0.2 * size_ratio), 3
    )
    severity_score = min(max(severity_score, 0.0), 1.0)

    if severity_score >= 0.75:
        label = "Critical"
    elif severity_score >= 0.45:
        label = "Moderate"
    else:
        label = "Minor"

    return {"severity_score": severity_score, "severity_label": label}


if __name__ == "__main__":
    # Quick test: python src/severity.py
    result = estimate_severity("crack", confidence=0.96, area_cm2=4.2)
    print(result)
