"""
src/rul_model.py
LSTM-based Remaining Useful Life (RUL) predictor for aircraft components.
Falls back to a simple heuristic if no trained model/scaler is available,
so the rest of the pipeline can run before training is complete.
"""

import os
import random

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "rul"
)


class RULPredictor:
    """LSTM model for predicting Remaining Useful Life from sensor sequences."""

    def __init__(self, input_size=14, hidden_size=64, num_layers=2):
        import torch.nn as nn

        class _LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=0.2
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        self.net = _LSTMNet()

    def forward(self, x):
        return self.net(x)


def load_trained_model():
    """Load a trained RUL model + scaler if present, else return None."""
    model_path = os.path.join(MODEL_DIR, "rul_lstm.pt")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        import torch
        import joblib
        model = RULPredictor()
        model.net.load_state_dict(torch.load(model_path))
        model.net.eval()
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None


def predict_rul(component="APU", sensor_df=None):
    """
    Predict RUL (in days) for a given component.

    If a trained model exists in models/rul/, use it. Otherwise,
    fall back to a sensible mock value so the rest of the app works
    during early development.

    Args:
        component (str): component name
        sensor_df (pd.DataFrame): recent sensor readings (optional, required for real model)

    Returns:
        int: predicted RUL in days
    """
    model, scaler = load_trained_model()

    if model is None or sensor_df is None:
        # Mock RUL values per component, until LSTM is trained
        mock_rul = {
            "APU": 7,
            "LandingGear": 18,
            "Hydraulics": 55,
            "Engine": 82,
            "FuselageSkin": 31,
        }
        base = mock_rul.get(component, 40)
        return max(1, base + random.randint(-3, 3))

    # Real inference path (once model + scaler exist)
    import torch
    feature_cols = ["vibration", "temperature", "pressure"]
    scaled = scaler.transform(sensor_df[feature_cols].values)
    tensor = torch.FloatTensor(scaled).unsqueeze(0)
    with torch.no_grad():
        rul = model.forward(tensor).item()
    return max(0, round(rul))


if __name__ == "__main__":
    # Quick test: python src/rul_model.py
    print(predict_rul("APU"))
