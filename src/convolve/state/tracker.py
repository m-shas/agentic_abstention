from __future__ import annotations

import uuid
from datetime import UTC, datetime

from convolve.state.store import StateStore
from convolve.types import (
    Decision,
    Reason,
    Step,
    Trajectory,
)


def new_session_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class StateTracker:
    """Owns the lifecycle of an in-flight trajectory.

    Responsibilities
    ----------------
    * create / load / persist trajectories
    * append proposed actions and reported observations
    * keep the cumulative cost accurate
    """

    def __init__(self, store: StateStore) -> None:
        self._store = store

    async def create(
        self,
        task: str,
        max_steps: int,
        budget: float,
        threshold: float,
    ) -> Trajectory:
        traj = Trajectory(
            session_id=new_session_id(),
            task=task,
            max_steps=max_steps,
            budget=budget,
            threshold=threshold,
        )
        await self._store.put(traj)
        return traj

    async def load(self, session_id: str) -> Trajectory | None:
        return await self._store.get(session_id)

    async def save(self, traj: Trajectory) -> None:
        await self._store.put(traj)

    async def close(self, session_id: str) -> None:
        await self._store.delete(session_id)

    # -- mutations ---------------------------------------------------------

    def append_proposed(
        self,
        traj: Trajectory,
        thought: str,
        action: str,
        action_input: dict,
        estimated_cost: float,
        stop_probability: float,
        threshold_used: float,
        base_threshold: float,
        reason: Reason,
    ) -> Step:
        step = Step(
            idx=traj.n_steps,
            thought=thought,
            action=action,
            action_input=action_input,
            estimated_cost=estimated_cost,
            stop_probability=stop_probability,
            threshold_used=threshold_used,
            decision=Decision.ABSTAIN if reason != Reason.UNDER_THRESHOLD else Decision.CONTINUE,
            reason=reason,
        )
        traj.steps.append(step)
        return step

    def attach_observation(
        self,
        traj: Trajectory,
        observation: str,
        cost: float,
        is_error: bool | None,
    ) -> Step:
        if not traj.steps:
            raise ValueError("cannot attach observation before any step was proposed")
        step = traj.steps[-1]
        step.observation = observation
        step.actual_cost = float(cost)
        step.is_error = bool(is_error) if is_error is not None else looks_like_error(observation)
        traj.spent = round(traj.spent + float(cost), 6)
        return step

    def pending_step(self, traj: Trajectory) -> Step | None:
        """The most recent step that has no observation yet, if any."""
        if traj.steps and traj.steps[-1].observation is None:
            return traj.steps[-1]
        return None


_ERROR_MARKERS = (
    "error",
    "failed",
    "failure",
    "not found",
    "no results",
    "invalid",
    "unavailable",
    "unable",
    "cannot",
    "can not",
    "does not exist",
    "denied",
    "forbidden",
    "timeout",
    "timed out",
    "exception",
    "traceback",
    "no matching",
    "empty result",
    "out of stock",
    "not available",
)


def looks_like_error(observation: str) -> bool:
    low = observation.lower()
    return any(marker in low for marker in _ERROR_MARKERS)
