from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Standard ECE with equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / y_true.size) * abs(acc - conf)
    return float(ece)


@dataclass
class IsotonicCalibrator:
    """Monotonic, non-parametric calibrator. Preferred over temperature scaling
    when the base scores are already well-separated but mis-scaled."""

    model: IsotonicRegression | None = None
    fitted: bool = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        y_prob = np.asarray(y_prob, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if len(np.unique(y_true)) < 2:
            self.model = None
            self.fitted = False
            return self
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(y_prob, y_true)
        self.model = iso
        self.fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        y_prob = np.asarray(y_prob, dtype=float)
        if not self.fitted or self.model is None:
            return np.clip(y_prob, 0.0, 1.0)
        return np.clip(self.model.predict(y_prob), 0.0, 1.0)

    def transform_one(self, p: float) -> float:
        return float(self.transform(np.asarray([p], dtype=float))[0])
