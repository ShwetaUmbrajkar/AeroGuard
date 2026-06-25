"""
src/detection.py
Wrapper around a trained YOLOv8 model for aircraft defect detection.
Falls back to mock detections if no trained model is available yet,
so the rest of the pipeline (dashboard, fusion) can be developed
and demoed before training finishes.
"""

import os
import random

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "defect_detection", "best.pt"
)

_model = None


def _load_model():
    """Lazily load the YOLO model only when needed."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            if os.path.exists(MODEL_PATH):
                _model = YOLO(MODEL_PATH)
            else:
                _model = "mock"
        except ImportError:
            _model = "mock"
    return _model


def detect_defects(image_path):
    """
    Run defect detection on an image.

    Args:
        image_path (str): path to the inspection image

    Returns:
        list of dicts: [{class, confidence, area_cm2, bbox}, ...]
    """
    model = _load_model()

    if model == "mock":
        # No trained weights yet — return realistic mock detections
        # so the dashboard and fusion layer can still be tested.
        mock_defects = [
            {"class": "crack", "confidence": 0.96, "area_cm2": 4.2, "bbox": [120, 80, 180, 140]},
            {"class": "corrosion", "confidence": 0.88, "area_cm2": 9.7, "bbox": [300, 200, 380, 260]},
            {"class": "dent", "confidence": 0.79, "area_cm2": 2.1, "bbox": [50, 300, 90, 340]},
        ]
        return random.sample(mock_defects, k=random.randint(1, len(mock_defects)))

    # Real inference once best.pt is trained and placed in models/defect_detection/
    results = model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            width_px = xyxy[2] - xyxy[0]
            height_px = xyxy[3] - xyxy[1]
            # NOTE: replace this pixel->cm2 conversion with a calibrated
            # value once you know your camera's real-world scale.
            area_cm2 = round((width_px * height_px) / 1000, 2)
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "area_cm2": area_cm2,
                "bbox": xyxy,
            })
    return detections


if __name__ == "__main__":
    # Quick test: python src/detection.py
    print(detect_defects("data/raw/sample.jpg"))
