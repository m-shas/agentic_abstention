from __future__ import annotations

import asyncio
import json
import time
from typing import Protocol

from convolve.types import Trajectory


class StateStore(Protocol):
    """Hot storage for in-flight trajectories."""

    async def put(self, traj: Trajectory) -> None: ...
    async def get(self, session_id: str) -> Trajectory | None: ...
    async def delete(self, session_id: str) -> None: ...
    async def close(self) -> None: ...
    def backend_name(self) -> str: ...


def _dump(traj: Trajectory) -> str:
    return json.dumps(
        {
            "session_id": traj.session_id,
            "task": traj.task,
            "max_steps": traj.max_steps,
            "budget": traj.budget,
            "threshold": traj.threshold,
            "spent": traj.spent,
            "steps": [
                {
                    "idx": s.idx,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "estimated_cost": s.estimated_cost,
                    "actual_cost": s.actual_cost,
                    "is_error": s.is_error,
                    "stop_probability": s.stop_probability,
                    "threshold_used": s.threshold_used,
                    "decision": s.decision.value if s.decision else None,
                    "reason": s.reason.value if s.reason else None,
                }
                for s in traj.steps
            ],
        },
        ensure_ascii=False,
    )


def _load(raw: str | bytes) -> Trajectory:
    from convolve.types import Decision, Reason, Step

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    d = json.loads(raw)
    steps: list[Step] = []
    for s in d["steps"]:
        steps.append(
            Step(
                idx=s["idx"],
                thought=s["thought"],
                action=s["action"],
                action_input=s["action_input"],
                observation=s["observation"],
                estimated_cost=s["estimated_cost"],
                actual_cost=s["actual_cost"],
                is_error=s["is_error"],
                stop_probability=s["stop_probability"],
                threshold_used=s["threshold_used"],
                decision=Decision(s["decision"]) if s["decision"] else None,
                reason=Reason(s["reason"]) if s["reason"] else None,
            )
        )
    return Trajectory(
        session_id=d["session_id"],
        task=d["task"],
        max_steps=d["max_steps"],
        budget=d["budget"],
        threshold=d["threshold"],
        spent=d["spent"],
        steps=steps,
    )


class InMemoryStateStore:
    """Process-local store. Used when CONVOLVE_REDIS_URL is empty and in tests."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._data: dict[str, tuple[str, float]] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def put(self, traj: Trajectory) -> None:
        async with self._lock:
            self._data[traj.session_id] = (_dump(traj), time.monotonic() + self._ttl)

    async def get(self, session_id: str) -> Trajectory | None:
        async with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return None
            raw, expires = entry
            if expires < time.monotonic():
                self._data.pop(session_id, None)
                return None
            return _load(raw)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._data.pop(session_id, None)

    async def close(self) -> None:
        async with self._lock:
            self._data.clear()

    def backend_name(self) -> str:
        return "memory"


class RedisStateStore:
    """Redis-backed store. One JSON blob per session with a TTL."""

    PREFIX = "convolve:traj:"

    def __init__(self, url: str, ttl_seconds: int = 3600) -> None:
        from redis.asyncio import Redis  # imported lazily

        self._redis: Redis = Redis.from_url(url, decode_responses=False)
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{self.PREFIX}{session_id}"

    async def put(self, traj: Trajectory) -> None:
        await self._redis.set(self._key(traj.session_id), _dump(traj), ex=self._ttl)

    async def get(self, session_id: str) -> Trajectory | None:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        return _load(raw)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    async def close(self) -> None:
        await self._redis.aclose()

    def backend_name(self) -> str:
        return "redis"
