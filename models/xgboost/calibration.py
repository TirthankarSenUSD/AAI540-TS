"""A minimal, dependency-free Platt-scaling calibrator.

`sklearn.calibration.CalibratedClassifierCV` (however it's constructed —
`cv="prefit"` or the modern `FrozenEstimator` wrapper) pickles the *entire*
wrapped estimator plus sklearn-internal state into the calibrator artifact.
That's two problems for a deployed artifact: it duplicates the whole model
a second time inside calibrator.joblib, and joblib/pickle compatibility
across sklearn versions between the dev environment and the serving
container is not guaranteed — training/serving skew for the calibrator
itself, which is exactly the fragility Task 4's smoke test exists to catch.

Platt scaling is just a 1-D logistic regression on the model's raw score:
`p_calibrated = sigmoid(a * raw_score + b)`. Storing only the two fitted
floats sidesteps both problems — no sklearn object to unpickle at serving
time at all.
"""
from __future__ import annotations

import numpy as np


class PlattCalibrator:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def predict_proba(self, raw_scores) -> np.ndarray:
        raw_scores = np.asarray(raw_scores, dtype=float)
        p1 = 1.0 / (1.0 + np.exp(-(self.a * raw_scores + self.b)))
        return np.column_stack([1 - p1, p1])
