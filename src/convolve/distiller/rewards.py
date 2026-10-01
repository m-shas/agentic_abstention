from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convolve.types import Outcome

# Terminal reward per session outcome.
OUTCOME_REWARDS: dict[str, float] = {
    Outcome.SUCCESS.value: 1.0,
    Outcome.ABSTAINED.value: 0.5,
    Outcome.ERROR.value: -0.5,
    Outcome.ABORTED.value: -0.3,
    Outcome.LOOPED.value: -1.0,
}

# Per-step penalty applied to return-to-go, discouraging long futile runs.
STEP_COST_PENALTY = 0.05


@dataclass(slots=True)
class RewardConfig:
    outcome_rewards: dict[str, float] | None = None
    step_cost_penalty: float = STEP_COST_PENALTY

    def __post_init__(self) -> None:
        if self.outcome_rewards is None:
            self.outcome_rewards = dict(OUTCOME_REWARDS)


def outcome_reward(outcome: str, cfg: RewardConfig | None = None) -> float:
    cfg = cfg or RewardConfig()
    rewards = cfg.outcome_rewards or OUTCOME_REWARDS
    return float(rewards.get(outcome, -0.5))


def return_to_go(
    n_steps: int,
    terminal_reward: float,
    step_cost_penalty: float = STEP_COST_PENALTY,
) -> np.ndarray:
    """Compute per-step return-to-go for a trajectory of length `n_steps`.

    ``rtg[t] = terminal_reward - step_cost_penalty * (n_steps - 1 - t)``

    This is monotonically increasing in ``t``: earlier steps have lower RTG
    because more cost remains ahead.
    """
    if n_steps <= 0:
        return np.zeros(0, dtype=np.float32)
    offsets = np.arange(n_steps - 1, -1, -1, dtype=np.float32)
    return (terminal_reward - step_cost_penalty * offsets).astype(np.float32)
