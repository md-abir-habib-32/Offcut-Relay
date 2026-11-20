"""Evaluate adapting pickup probabilities to a simulated change in conditions."""
import json
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def synthetic_pickups(seed: int, size: int, shifted: bool):
    rng = np.random.default_rng(seed)
    features = rng.uniform([1 / 20, 0, 8 / 45], [26 / 20, 1, 46 / 45], size=(size, 3))
    # The shift represents a lower rush-period pickup rate and a greater distance penalty.
    intercept, weights = (-0.1, [-4.1, 3.8, -0.45]) if shifted else (1.05, [-3.2, 3.8, -0.45])
    probability = 1 / (1 + np.exp(-(intercept + features @ weights)))
    labels = (rng.random(size) < probability).astype(int)
    return features, labels


def evaluate():
    source_x, source_y = synthetic_pickups(2026, 700, False)
    calibration_x, calibration_y = synthetic_pickups(5331, 500, True)
    test_x, test_y = synthetic_pickups(8321, 500, True)
    base_model = LogisticRegression(C=10).fit(source_x, source_y)
    adapted_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    adapted_model.fit(calibration_x, calibration_y)

    def scores(model):
        prediction = model.predict_proba(test_x)[:, 1]
        return {
            "brier": round(float(brier_score_loss(test_y, prediction)), 4),
            "log_loss": round(float(log_loss(test_y, prediction)), 4),
        }

    return {
        "data": "synthetic",
        "features": ["distance", "pickup slack", "order weight"],
        "source_training_rows": len(source_y),
        "target_calibration_rows": len(calibration_y),
        "target_holdout_rows": len(test_y),
        "baseline": scores(base_model),
        "adapted": scores(adapted_model),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
