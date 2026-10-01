from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from convolve.distiller.dataset import (
    BOS,
    build_dataset,
    encode_trajectory,
    load_trajectories,
    tokenize,
)
from convolve.distiller.rewards import (
    OUTCOME_REWARDS,
    RewardConfig,
    outcome_reward,
    return_to_go,
)


def _record(
    outcome: str, n_steps: int, is_error: bool = False, session_id: str | None = None
) -> dict:
    return {
        "session_id": session_id or f"s-{outcome}-{n_steps}",
        "task": "book a flight under Rs.500",
        "max_steps": 12,
        "budget": 0.5,
        "threshold": 0.7,
        "spent": 0.05 * n_steps,
        "outcome": outcome,
        "steps": [
            {
                "idx": i,
                "thought": f"attempt {i}",
                "action": "search_flights",
                "action_input": {"i": i, "max_price": 500},
                "observation": "no results found" if is_error else "OK",
                "estimated_cost": 0.05,
                "actual_cost": 0.05,
                "is_error": is_error,
                "stop_probability": 0.3 + 0.05 * i,
                "threshold_used": 0.7,
                "decision": "continue",
                "reason": "under_threshold",
            }
            for i in range(n_steps)
        ],
    }


def test_outcome_reward_lookup():
    assert outcome_reward("success") == OUTCOME_REWARDS["success"]
    assert outcome_reward("looped") == OUTCOME_REWARDS["looped"]
    assert outcome_reward("unknown") == -0.5


def test_return_to_go_shape_and_monotonicity():
    rtg = return_to_go(5, terminal_reward=1.0, step_cost_penalty=0.1)
    assert rtg.shape == (5,)
    assert np.all(np.diff(rtg) > 0)
    assert rtg[-1] == pytest.approx(1.0)
    assert rtg[0] == pytest.approx(1.0 - 0.1 * 4)


def test_return_to_go_empty():
    assert return_to_go(0, 1.0).shape == (0,)


def test_tokenize_deterministic():
    a = tokenize("Book a flight under 500")
    b = tokenize("Book a flight under 500")
    assert a == b
    assert all(isinstance(x, int) for x in a)


def test_encode_trajectory_produces_sample():
    rec = _record("looped", 5, is_error=True)
    sample = encode_trajectory(rec)
    assert sample.tokens[0] == BOS
    assert sample.tokens.shape[0] == sample.rtg.shape[0]
    assert sample.tokens.shape[0] == sample.action_targets.shape[0]
    assert sample.outcome == "looped"


def test_build_dataset_pads_correctly():
    records = [_record("looped", 3, True), _record("success", 2, False)]
    ds = build_dataset(records, max_len=64)
    assert ds.tokens.shape == (2, 64)
    assert ds.mask.shape == (2, 64)
    assert ds.mask[0].sum() > 0
    # Padding is zero
    assert (ds.tokens[0][ds.mask[0] == 0] == 0).all()


def test_train_val_split_disjoint():
    records = [_record("looped", 3, True, session_id=f"s-{i}") for i in range(10)]
    ds = build_dataset(records, max_len=32)
    train, val = ds.train_val_split(val_frac=0.2, seed=0)
    assert len(train) + len(val) == len(ds)
    assert len(val) >= 1
    train_ids = set(train.session_ids)
    val_ids = set(val.session_ids)
    assert train_ids.isdisjoint(val_ids)


def test_load_trajectories_roundtrip(tmp_path: Path):
    import json

    path = tmp_path / "traj.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for _ in range(3):
            fh.write(json.dumps(_record("looped", 2, True)) + "\n")
    loaded = load_trajectories(path)
    assert len(loaded) == 3
    assert all("steps" in r for r in loaded)


def test_load_trajectories_missing_file():
    assert load_trajectories(Path("/nonexistent/traj.jsonl")) == []


def test_reward_config_custom():
    cfg = RewardConfig(outcome_rewards={"success": 10.0}, step_cost_penalty=0.0)
    assert outcome_reward("success", cfg) == 10.0
    assert outcome_reward("looped", cfg) == -0.5
