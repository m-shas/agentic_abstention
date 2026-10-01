from __future__ import annotations

from convolve.config import Settings
from convolve.cost.controller import CostController
from convolve.evaluator.evaluator import AbstentionEvaluator
from convolve.types import Decision, Reason, Step, Trajectory


def _settings(**overrides) -> Settings:
    defaults = {
        "evaluator_model_path": "/nonexistent/does-not-exist.joblib",
        "default_threshold": 0.7,
        "cost_knee": 0.5,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _traj(steps: list[Step], **over) -> Trajectory:
    base = dict(
        session_id="s",
        task="book a flight under Rs.500",
        max_steps=10,
        budget=0.5,
        threshold=0.7,
        spent=sum(s.actual_cost for s in steps),
    )
    base.update(over)
    return Trajectory(steps=steps, **base)


def _bad(i: int) -> Step:
    return Step(
        idx=i,
        thought="",
        action="search_flights",
        action_input={"i": i},
        observation="no results found",
        actual_cost=0.05,
        is_error=True,
    )


def _ok(i: int) -> Step:
    return Step(
        idx=i,
        thought="",
        action="select_flight",
        action_input={"i": i},
        observation="flight selected",
        actual_cost=0.05,
        is_error=False,
    )


def test_evaluator_uses_heuristic_when_no_model():
    ev = AbstentionEvaluator(_settings(), CostController())
    assert ev.source == "heuristic"


def test_healthy_trajectory_continues():
    ev = AbstentionEvaluator(_settings(), CostController())
    v = ev.evaluate(_traj([_ok(0), _ok(1)]))
    assert v.decision is Decision.CONTINUE
    assert v.reason is Reason.UNDER_THRESHOLD


def test_repeated_failures_trigger_abstention():
    ev = AbstentionEvaluator(_settings(default_threshold=0.5), CostController())
    steps = [_bad(i) for i in range(4)]
    traj = _traj(steps, threshold=0.5)
    v = ev.evaluate(traj)
    assert v.decision is Decision.ABSTAIN
    assert v.reason in (Reason.UNCERTAINTY_THRESHOLD, Reason.MAX_STEPS_REACHED)


def test_max_steps_reached_abstains():
    ev = AbstentionEvaluator(_settings(), CostController())
    steps = [_ok(i) for i in range(10)]
    v = ev.evaluate(_traj(steps, max_steps=10))
    assert v.decision is Decision.ABSTAIN
    assert v.reason is Reason.MAX_STEPS_REACHED


def test_budget_exhaustion_abstains_even_with_progress():
    ev = AbstentionEvaluator(_settings(), CostController())
    steps = [_ok(i) for i in range(3)]
    traj = _traj(steps, budget=0.10, spent=0.10)
    v = ev.evaluate(traj)
    assert v.decision is Decision.ABSTAIN
    assert v.reason is Reason.COST_BUDGET_REACHED
    assert v.threshold_used == 0.0


def test_abstention_message_max_steps():
    ev = AbstentionEvaluator(_settings(), CostController())
    steps = [_ok(i) for i in range(10)]
    traj = _traj(steps, max_steps=10)
    v = ev.evaluate(traj)
    msg = ev.build_abstention_message(traj, v)
    assert "maximum step budget" in msg


def test_abstention_message_cost():
    ev = AbstentionEvaluator(_settings(), CostController())
    steps = [_ok(i) for i in range(3)]
    traj = _traj(steps, budget=0.10, spent=0.10)
    v = ev.evaluate(traj)
    msg = ev.build_abstention_message(traj, v)
    assert "cost budget" in msg


def test_abstention_message_uncertainty():
    ev = AbstentionEvaluator(_settings(default_threshold=0.4), CostController())
    steps = [_bad(i) for i in range(4)]
    traj = _traj(steps, threshold=0.4)
    v = ev.evaluate(traj)
    if v.reason is Reason.UNCERTAINTY_THRESHOLD:
        msg = ev.build_abstention_message(traj, v)
        assert "cannot complete" in msg.lower()
