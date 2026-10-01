from __future__ import annotations

import numpy as np

from convolve.evaluator.calibration import (
    IsotonicCalibrator,
    expected_calibration_error,
)


def test_ece_zero_on_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error(y_true, y_prob, n_bins=4)
    assert ece == 0.0


def test_ece_high_on_inverted_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.9, 0.1, 0.1])
    ece = expected_calibration_error(y_true, y_prob, n_bins=4)
    assert ece > 0.5


def test_ece_handles_empty_input():
    ece = expected_calibration_error(np.array([]), np.array([]))
    assert ece == 0.0


def test_isotonic_reduces_ece():
    rng = np.random.default_rng(0)
    n = 500
    raw = rng.uniform(0, 1, size=n).astype(np.float32)
    # True probability is not equal to raw; it is skewed.
    y = (rng.uniform(0, 1, size=n) < (raw**2)).astype(int)
    ece_before = expected_calibration_error(y, raw)
    cal = IsotonicCalibrator().fit(raw, y)
    calibrated = cal.transform(raw)
    ece_after = expected_calibration_error(y, calibrated)
    assert ece_after <= ece_before + 1e-6


def test_isotonic_identity_when_unfitted():
    cal = IsotonicCalibrator()
    out = cal.transform(np.array([0.2, 0.8]))
    assert np.allclose(out, [0.2, 0.8])


def test_isotonic_handles_single_class():
    cal = IsotonicCalibrator().fit(np.array([0.1, 0.5, 0.9]), np.array([1, 1, 1]))
    out = cal.transform(np.array([0.3, 0.7]))
    # Falls back to identity when training labels have one class.
    assert np.allclose(out, [0.3, 0.7])


def test_isotonic_clips_output():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, size=50).astype(np.float32)
    y = (raw > 0.5).astype(int)
    cal = IsotonicCalibrator().fit(raw, y)
    out = cal.transform(np.array([-0.5, 1.5]))
    assert out.min() >= 0.0
    assert out.max() <= 1.0
