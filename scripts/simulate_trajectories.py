"""Synthetic trajectory generator.

Produces labelled trajectories that mirror the patterns the real evaluator
and distiller must learn:

* ``success``   — an agent that walks a solvable task to completion.
* ``abstained`` — an agent that stops as soon as it recognises impossibility.
* ``looped``    — an agent that keeps trying an impossible task until max steps.

The generator is deterministic given a seed and requires no external services.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class _TaskTemplate:
    task: str
    tools: tuple[str, ...]
    solvable: bool
    failure_reason: str = ""
    cost_per_call: float = 0.05


TEMPLATES: tuple[_TaskTemplate, ...] = (
    _TaskTemplate(
        "Book a flight from BOM to DEL under Rs.500",
        ("search_flights", "select_flight", "make_payment"),
        solvable=False,
        failure_reason="No flights matching criteria. Cheapest available direct flight is Rs.4,500.",
    ),
    _TaskTemplate(
        "Book a flight from BOM to DEL on 2026-08-15",
        ("search_flights", "select_flight", "make_payment"),
        solvable=True,
    ),
    _TaskTemplate(
        "Apply coupon FREE100 to get a 100% discount on a MacBook Pro",
        ("validate_coupon", "apply_coupon", "checkout"),
        solvable=False,
        failure_reason="Coupon not valid for electronics.",
    ),
    _TaskTemplate(
        "Apply coupon SAVE5 to this order",
        ("validate_coupon", "apply_coupon", "checkout"),
        solvable=True,
    ),
    _TaskTemplate(
        "Generate a sales report for the Q3_2020_legacy table",
        ("list_tables", "describe_table", "run_query"),
        solvable=False,
        failure_reason="Table 'Q3_2020_legacy' does not exist.",
    ),
    _TaskTemplate(
        "Generate a sales report for the sales_2026 table",
        ("list_tables", "describe_table", "run_query"),
        solvable=True,
    ),
    _TaskTemplate(
        "Find the clause in the Indian Constitution allowing the President to dissolve the Supreme Court",
        ("search_legal_db", "lookup_article", "cite_case"),
        solvable=False,
        failure_reason="No such clause found. Supreme Court independence is part of the basic structure doctrine.",
    ),
    _TaskTemplate(
        "Find the clause in the Indian Constitution on the right to equality",
        ("search_legal_db", "lookup_article", "cite_case"),
        solvable=True,
    ),
    _TaskTemplate(
        "Turn on the garden fountain lights to blue at 50% brightness",
        ("list_devices", "set_device_state", "set_brightness"),
        solvable=False,
        failure_reason="No 'garden fountain lights' device registered.",
    ),
    _TaskTemplate(
        "Turn on the living room lights",
        ("list_devices", "set_device_state", "set_brightness"),
        solvable=True,
    ),
)


@dataclass
class _Step:
    idx: int = 0
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str | None = None
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    is_error: bool = False
    stop_probability: float | None = None
    threshold_used: float | None = None
    decision: str | None = None
    reason: str | None = None


@dataclass
class _Trajectory:
    session_id: str
    task: str
    max_steps: int
    budget: float
    threshold: float
    spent: float = 0.0
    outcome: str = "error"
    final_answer: str | None = None
    steps: list[_Step] = field(default_factory=list)
    ended_at: str = ""


def _succ_trajectory(
    tpl: _TaskTemplate, rng: random.Random, sid: str, threshold: float
) -> _Trajectory:
    steps: list[_Step] = []
    spent = 0.0
    n = len(tpl.tools)
    for i, tool in enumerate(tpl.tools):
        is_last = i == n - 1
        obs = "OK" if not is_last else "Completed successfully."
        cost = tpl.cost_per_call
        spent = round(spent + cost, 6)
        step = _Step(
            idx=i,
            thought=f"Step {i + 1}: call {tool}",
            action=tool,
            action_input={"arg": f"value_{i}"},
            observation=obs,
            estimated_cost=cost,
            actual_cost=cost,
            is_error=False,
            stop_probability=round(0.05 + 0.01 * i, 4),
            threshold_used=threshold,
            decision="continue",
            reason="under_threshold",
        )
        steps.append(step)
    return _Trajectory(
        session_id=sid,
        task=tpl.task,
        max_steps=12,
        budget=0.5,
        threshold=threshold,
        spent=spent,
        outcome="success",
        final_answer="Task completed.",
        steps=steps,
        ended_at=datetime.now(timezone.utc).isoformat(),
    )


def _abstain_trajectory(
    tpl: _TaskTemplate, rng: random.Random, sid: str, threshold: float
) -> _Trajectory:
    steps: list[_Step] = []
    spent = 0.0
    n_try = rng.randint(2, 4)
    for i in range(n_try):
        tool = tpl.tools[i % len(tpl.tools)]
        is_last = i == n_try - 1
        obs = tpl.failure_reason if is_last else f"Attempt {i + 1}: {tpl.failure_reason}"
        cost = tpl.cost_per_call
        spent = round(spent + cost, 6)
        prob = min(0.95, 0.2 + 0.25 * i)
        decision = "abstain" if is_last else "continue"
        reason = "uncertainty_threshold" if is_last else "under_threshold"
        step = _Step(
            idx=i,
            thought=f"Attempt {i + 1} via {tool}",
            action=tool,
            action_input={"attempt": i, "arg": f"value_{i}"},
            observation=obs,
            estimated_cost=cost,
            actual_cost=cost,
            is_error=True,
            stop_probability=round(prob, 4),
            threshold_used=threshold,
            decision=decision,
            reason=reason,
        )
        steps.append(step)
    return _Trajectory(
        session_id=sid,
        task=tpl.task,
        max_steps=12,
        budget=0.5,
        threshold=threshold,
        spent=spent,
        outcome="abstained",
        final_answer="Cannot complete this task.",
        steps=steps,
        ended_at=datetime.now(timezone.utc).isoformat(),
    )


def _looped_trajectory(
    tpl: _TaskTemplate, rng: random.Random, sid: str, threshold: float
) -> _Trajectory:
    steps: list[_Step] = []
    spent = 0.0
    n = 10
    for i in range(n):
        tool = tpl.tools[i % len(tpl.tools)]
        cost = tpl.cost_per_call
        spent = round(spent + cost, 6)
        step = _Step(
            idx=i,
            thought=f"Attempt {i + 1}: maybe try {tool}",
            action=tool,
            action_input={"attempt": i, "small_variation": i},
            observation=tpl.failure_reason,
            estimated_cost=cost,
            actual_cost=cost,
            is_error=True,
            stop_probability=round(min(0.65, 0.3 + 0.03 * i), 4),
            threshold_used=threshold,
            decision="continue",
            reason="under_threshold",
        )
        steps.append(step)
    return _Trajectory(
        session_id=sid,
        task=tpl.task,
        max_steps=12,
        budget=0.5,
        threshold=threshold,
        spent=spent,
        outcome="looped",
        final_answer=None,
        steps=steps,
        ended_at=datetime.now(timezone.utc).isoformat(),
    )


def _to_dict(t: _Trajectory) -> dict:
    return {
        "session_id": t.session_id,
        "task": t.task,
        "max_steps": t.max_steps,
        "budget": t.budget,
        "threshold": t.threshold,
        "spent": t.spent,
        "outcome": t.outcome,
        "final_answer": t.final_answer,
        "ended_at": t.ended_at,
        "steps": [s.__dict__ for s in t.steps],
    }


def _iter_trajectories(n: int, seed: int, threshold: float = 0.7) -> Iterable[_Trajectory]:
    rng = random.Random(seed)
    for i in range(n):
        tpl = TEMPLATES[i % len(TEMPLATES)]
        sid = f"sim-{seed}-{i:06d}"
        if tpl.solvable:
            yield _succ_trajectory(tpl, rng, sid, threshold)
        else:
            roll = rng.random()
            if roll < 0.35:
                yield _abstain_trajectory(tpl, rng, sid, threshold)
            elif roll < 0.70:
                yield _looped_trajectory(tpl, rng, sid, threshold)
            else:
                yield _abstain_trajectory(tpl, rng, sid, threshold)


def generate(n: int, seed: int, path: Path, threshold: float = 0.7) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as fh:
        for t in _iter_trajectories(n, seed, threshold):
            fh.write(json.dumps(_to_dict(t), ensure_ascii=False) + "\n")
            written += 1
    return written


def _cli() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Generate synthetic Convolve trajectories.")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument(
        "--out", type=Path, default=Path("data/trajectories/trajectories.jsonl")
    )
    args = p.parse_args()
    n = generate(args.n, args.seed, args.out, args.threshold)
    print(f"Wrote {n} trajectories to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
