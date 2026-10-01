from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Decision(StrEnum):
    CONTINUE = "continue"
    ABSTAIN = "abstain"


class SessionStatus(StrEnum):
    OPEN = "open"
    ENDED = "ended"


class Outcome(StrEnum):
    SUCCESS = "success"
    ABSTAINED = "abstained"
    LOOPED = "looped"
    ERROR = "error"
    ABORTED = "aborted"


class Reason(StrEnum):
    UNDER_THRESHOLD = "under_threshold"
    UNCERTAINTY_THRESHOLD = "uncertainty_threshold"
    COST_BUDGET_REACHED = "cost_budget_reached"
    MAX_STEPS_REACHED = "max_steps_reached"


# --------------------------------------------------------------------------
# Internal (non-serialised) data structures
# --------------------------------------------------------------------------


def fingerprint(action: str, action_input: dict[str, Any]) -> str:
    payload = json.dumps(
        {"a": action, "i": action_input}, sort_keys=True, default=str, ensure_ascii=False
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class Step:
    idx: int
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str | None = None
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    is_error: bool = False
    stop_probability: float | None = None
    threshold_used: float | None = None
    decision: Decision | None = None
    reason: Reason | None = None

    def action_fingerprint(self) -> str:
        """Stable hash of action + normalised input, used for repeat detection."""
        return fingerprint(self.action, self.action_input)


@dataclass(slots=True)
class Trajectory:
    session_id: str
    task: str
    max_steps: int
    budget: float
    threshold: float
    steps: list[Step] = field(default_factory=list)
    spent: float = 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def resolved_steps(self) -> list[Step]:
        return [s for s in self.steps if s.observation is not None]


# --------------------------------------------------------------------------
# Serialisable payloads
# --------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)
    max_steps: int | None = Field(default=None, ge=1, le=200)
    budget: float | None = Field(default=None, ge=0.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    session_id: str
    task: str
    max_steps: int
    budget: float
    threshold: float
    created_at: str


class ProposeStepRequest(BaseModel):
    thought: str = Field(default="", max_length=8000)
    action: str = Field(min_length=1, max_length=128)
    action_input: dict[str, Any] = Field(default_factory=dict)
    estimated_cost: float = Field(default=0.0, ge=0.0)


class ProposeStepResponse(BaseModel):
    decision: Decision
    stop_probability: float
    threshold: float
    base_threshold: float
    reason: Reason
    step_index: int
    playbook_id: str | None = None
    abstention_message: str | None = None


class ObservationRequest(BaseModel):
    observation: str = Field(default="", max_length=100_000)
    cost: float = Field(default=0.0, ge=0.0)
    is_error: bool | None = None


class ObservationResponse(BaseModel):
    ok: bool
    step_index: int
    cumulative_cost: float


class EndSessionRequest(BaseModel):
    outcome: Outcome
    final_answer: str | None = Field(default=None, max_length=100_000)


class EndSessionResponse(BaseModel):
    ok: bool
    session_id: str
    outcome: Outcome
    total_steps: int
    total_cost: float
    trajectory_path: str | None


class SessionView(BaseModel):
    session_id: str
    task: str
    status: SessionStatus
    outcome: Outcome | None
    max_steps: int
    budget: float
    spent: float
    threshold: float
    steps: list[dict[str, Any]]


class PlaybookView(BaseModel):
    id: str
    name: str
    task_signature: str
    task_exemplar: str
    rule_text: str
    abstain_at_step: int
    score: float
    usage_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    state_backend: str
    evaluator_source: str
