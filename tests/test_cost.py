from __future__ import annotations

import pytest

from convolve.cost.controller import CostController


def test_knee_validation():
    with pytest.raises(ValueError):
        CostController(knee=1.0)
    with pytest.raises(ValueError):
        CostController(knee=-0.1)


def test_no_budget_leaves_threshold_unchanged():
    c = CostController(knee=0.5)
    thr, forced = c.adjust(base_threshold=0.7, spent=10.0, budget=0.0)
    assert thr == 0.7
    assert forced is False


def test_below_knee_unchanged():
    c = CostController(knee=0.5)
    thr, forced = c.adjust(base_threshold=0.7, spent=0.2, budget=1.0)
    assert thr == pytest.approx(0.7)
    assert forced is False


def test_at_budget_forces_zero():
    c = CostController(knee=0.5)
    thr, forced = c.adjust(base_threshold=0.7, spent=1.0, budget=1.0)
    assert thr == 0.0
    assert forced is True


def test_over_budget_forces_zero():
    c = CostController(knee=0.5)
    thr, forced = c.adjust(base_threshold=0.7, spent=2.0, budget=1.0)
    assert thr == 0.0
    assert forced is True


def test_midpoint_is_monotone():
    c = CostController(knee=0.5)
    thr_low, _ = c.adjust(0.7, spent=0.60, budget=1.0)
    thr_mid, _ = c.adjust(0.7, spent=0.75, budget=1.0)
    thr_high, _ = c.adjust(0.7, spent=0.90, budget=1.0)
    assert 0.0 < thr_high < thr_mid < thr_low < 0.7


def test_remaining_reports_inf_without_budget():
    c = CostController()
    assert c.remaining(1.0, 0.0) == float("inf")


def test_remaining_clamped():
    c = CostController()
    assert c.remaining(0.3, 0.5) == pytest.approx(0.2)
    assert c.remaining(0.9, 0.5) == 0.0
