from __future__ import annotations

import pytest

from convolve.state.store import InMemoryStateStore
from convolve.state.tracker import StateTracker, looks_like_error
from convolve.types import Decision, Reason


@pytest.fixture
def tracker() -> StateTracker:
    return StateTracker(InMemoryStateStore(ttl_seconds=60))


async def test_create_and_load(tracker: StateTracker):
    traj = await tracker.create("task", max_steps=5, budget=0.5, threshold=0.7)
    assert traj.session_id
    assert traj.task == "task"
    loaded = await tracker.load(traj.session_id)
    assert loaded is not None
    assert loaded.session_id == traj.session_id
    assert loaded.max_steps == 5


async def test_load_missing_returns_none(tracker: StateTracker):
    assert await tracker.load("nope") is None


async def test_append_proposed_and_observation(tracker: StateTracker):
    traj = await tracker.create("task", max_steps=5, budget=0.5, threshold=0.7)
    step = tracker.append_proposed(
        traj,
        thought="t",
        action="a",
        action_input={"x": 1},
        estimated_cost=0.05,
        stop_probability=0.1,
        threshold_used=0.7,
        base_threshold=0.7,
        reason=Reason.UNDER_THRESHOLD,
    )
    assert step.decision is Decision.CONTINUE
    assert step.observation is None
    tracker.attach_observation(traj, observation="ok", cost=0.05, is_error=False)
    assert step.observation == "ok"
    assert traj.spent == pytest.approx(0.05)


async def test_observation_without_step_raises(tracker: StateTracker):
    traj = await tracker.create("task", max_steps=5, budget=0.5, threshold=0.7)
    with pytest.raises(ValueError):
        tracker.attach_observation(traj, observation="x", cost=0.0, is_error=False)


async def test_pending_step(tracker: StateTracker):
    traj = await tracker.create("task", max_steps=5, budget=0.5, threshold=0.7)
    assert tracker.pending_step(traj) is None
    tracker.append_proposed(
        traj, "t", "a", {}, 0.0, 0.1, 0.7, 0.7, Reason.UNDER_THRESHOLD
    )
    assert tracker.pending_step(traj) is not None
    tracker.attach_observation(traj, "ok", 0.0, False)
    assert tracker.pending_step(traj) is None


async def test_close_removes(tracker: StateTracker):
    traj = await tracker.create("task", max_steps=5, budget=0.5, threshold=0.7)
    await tracker.close(traj.session_id)
    assert await tracker.load(traj.session_id) is None


def test_looks_like_error():
    assert looks_like_error("Error: not found")
    assert looks_like_error("no results found")
    assert looks_like_error("Request timed out")
    assert not looks_like_error("Success. PNR ABC123.")
    assert not looks_like_error("Your booking is confirmed.")
