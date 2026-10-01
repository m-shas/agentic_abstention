from __future__ import annotations

import numpy as np

from convolve.evaluator.features import (
    FEATURE_DIM,
    FEATURE_ORDER,
    extract_features,
    heuristic_stop_probability,
    to_matrix,
    to_vector,
)
from convolve.types import Step, Trajectory


def _traj(steps: list[Step], task: str = "book a flight") -> Trajectory:
    return Trajectory(
        session_id="s",
        task=task,
        max_steps=10,
        budget=0.5,
        threshold=0.7,
        spent=sum(s.actual_cost for s in steps),
        steps=steps,
    )


def _ok_step(
    i: int, action: str = "search_flights", action_input: dict | None = None
) -> Step:
    return Step(
        idx=i,
        thought="",
        action=action,
        action_input=action_input if action_input is not None else {"i": i},
        observation="found results",
        actual_cost=0.05,
        is_error=False,
    )


def _bad_step(i: int, action: str = "search_flights") -> Step:
    return Step(
        idx=i,
        thought="",
        action=action,
        action_input={"i": i},
        observation="no results found",
        actual_cost=0.05,
        is_error=True,
    )


def test_feature_dim_and_order():
    assert FEATURE_DIM == len(FEATURE_ORDER)
    assert FEATURE_DIM == 12


def test_features_on_empty_trajectory():
    feats = extract_features(_traj([]))
    assert set(feats.keys()) == set(FEATURE_ORDER)
    for k, v in feats.items():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


def test_features_increase_with_failures():
    clean = extract_features(_traj([_ok_step(0), _ok_step(1)]))
    failing = extract_features(_traj([_bad_step(0), _bad_step(1), _bad_step(2)]))
    assert failing["error_ratio"] > clean["error_ratio"]
    assert failing["failure_streak_norm"] > clean["failure_streak_norm"]


def test_repeat_detection():
    repeated = _traj(
        [
            _ok_step(0, action="search", action_input={"x": 1}),
            _ok_step(1, action="search", action_input={"x": 1}),
            _ok_step(2, action="search", action_input={"x": 1}),
        ]
    )
    diverse = _traj(
        [
            Step(0, "", "a", {"x": 1}, "ok", 0.0, 0.0, False),
            Step(1, "", "b", {"x": 1}, "ok", 0.0, 0.0, False),
            Step(2, "", "c", {"x": 1}, "ok", 0.0, 0.0, False),
        ]
    )
    rf = extract_features(repeated)
    df = extract_features(diverse)
    assert rf["repeat_action_ratio"] > df["repeat_action_ratio"]
    assert rf["last_action_is_repeat"] == 1.0
    assert df["last_action_is_repeat"] == 0.0


def test_cost_fraction_capped():
    steps = [_ok_step(i) for i in range(6)]
    traj = _traj(steps)
    traj.spent = 1.0  # over budget
    feats = extract_features(traj)
    assert feats["cost_fraction"] == 1.0


def test_vector_and_matrix_shapes():
    feats = extract_features(_traj([_ok_step(0)]))
    v = to_vector(feats)
    assert v.shape == (FEATURE_DIM,)
    m = to_matrix([feats, feats])
    assert m.shape == (2, FEATURE_DIM)


def test_heuristic_responds_to_signal():
    healthy = heuristic_stop_probability(extract_features(_traj([_ok_step(0)])))
    doomed = heuristic_stop_probability(
        extract_features(_traj([_bad_step(0), _bad_step(1), _bad_step(2)]))
    )
    assert 0.0 < healthy < 1.0
    assert 0.0 < doomed < 1.0
    assert doomed > healthy


def test_feature_values_are_finite():
    feats = extract_features(_traj([_bad_step(0)]))
    arr = np.asarray(list(feats.values()))
    assert np.all(np.isfinite(arr))
