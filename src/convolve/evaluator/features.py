from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from convolve.types import Step, Trajectory

# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------

FEATURE_ORDER: tuple[str, ...] = (
    "step_index_norm",
    "n_steps_norm",
    "error_ratio",
    "failure_streak_norm",
    "repeat_action_ratio",
    "last_action_is_repeat",
    "action_diversity",
    "distinct_tools_ratio",
    "cost_fraction",
    "burn_rate_norm",
    "mean_observation_len_norm",
    "goal_overlap",
)

FEATURE_DIM = len(FEATURE_ORDER)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an the is are was were be been to of in on for with and or not do does
    i you we they it this that my your our their at by from as if then than""".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _resolved(steps: list[Step]) -> list[Step]:
    return [s for s in steps if s.observation is not None]


def _failure_streak(steps: list[Step]) -> int:
    streak = 0
    for s in reversed(steps):
        if s.observation is None:
            continue
        if s.is_error:
            streak += 1
        else:
            break
    return streak


def _repeat_stats(steps: list[Step]) -> tuple[float, float, int]:
    prints = [s.action_fingerprint() for s in steps]
    if not prints:
        return 0.0, 0.0, 0
    counts = Counter(prints)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    repeat_ratio = repeats / len(prints)
    last_is_repeat = 1.0 if counts[prints[-1]] > 1 else 0.0
    return repeat_ratio, last_is_repeat, len(counts)


def _burn_rate_norm(traj: Trajectory) -> float:
    """Fraction of budget spent per step, normalised to [0, 1]."""
    if traj.budget <= 0 or traj.n_steps == 0:
        return 0.0
    rate = (traj.spent / traj.budget) / traj.n_steps
    # A rate of 0.25/step saturates the feature.
    return float(min(1.0, rate / 0.25))


def extract_features(traj: Trajectory) -> dict[str, float]:
    """Extract the 12 abstention features from a trajectory.

    All values are floats in [0, 1] so that a linear model and a tree model
    behave comparably. The function is pure and deterministic.
    """
    steps = traj.steps
    resolved = _resolved(steps)
    n = len(resolved)
    max_steps = max(1, traj.max_steps)

    step_index_norm = min(1.0, len(steps) / max_steps)
    n_steps_norm = min(1.0, n / max_steps)

    error_ratio = (sum(1 for s in resolved if s.is_error) / n) if n else 0.0

    streak = _failure_streak(resolved)
    failure_streak_norm = min(1.0, streak / 3.0)

    repeat_ratio, last_is_repeat, distinct_actions = _repeat_stats(steps)
    action_diversity = (distinct_actions / len(steps)) if steps else 0.0

    tool_names = {s.action for s in steps}
    distinct_tools_ratio = min(1.0, len(tool_names) / 4.0)

    cost_fraction = 0.0
    if traj.budget > 0:
        cost_fraction = min(1.0, traj.spent / traj.budget)

    burn_rate_norm = _burn_rate_norm(traj)

    if resolved:
        mean_len = sum(len(s.observation or "") for s in resolved) / len(resolved)
        # ~600 chars saturates.
        mean_obs_len_norm = min(1.0, mean_len / 600.0)
    else:
        mean_obs_len_norm = 0.0

    task_tokens = set(_tokens(traj.task))
    obs_tokens: set[str] = set()
    for s in resolved:
        obs_tokens |= set(_tokens(s.observation or ""))
    goal_overlap = _jaccard(task_tokens, obs_tokens)

    return {
        "step_index_norm": float(step_index_norm),
        "n_steps_norm": float(n_steps_norm),
        "error_ratio": float(error_ratio),
        "failure_streak_norm": float(failure_streak_norm),
        "repeat_action_ratio": float(repeat_ratio),
        "last_action_is_repeat": float(last_is_repeat),
        "action_diversity": float(action_diversity),
        "distinct_tools_ratio": float(distinct_tools_ratio),
        "cost_fraction": float(cost_fraction),
        "burn_rate_norm": float(burn_rate_norm),
        "mean_observation_len_norm": float(mean_obs_len_norm),
        "goal_overlap": float(goal_overlap),
    }


def to_vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([features[k] for k in FEATURE_ORDER], dtype=np.float32)


def to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.vstack([to_vector(r) for r in rows]).astype(np.float32)


# ---------------------------------------------------------------------------
# Heuristic scorer
# ---------------------------------------------------------------------------

HEURISTIC_WEIGHTS: dict[str, float] = {
    "error_ratio": 0.28,
    "failure_streak_norm": 0.22,
    "repeat_action_ratio": 0.20,
    "last_action_is_repeat": 0.10,
    "burn_rate_norm": 0.08,
    "step_index_norm": 0.06,
    "action_diversity": -0.10,
    "goal_overlap": -0.06,
}

_HEURISTIC_PIVOT = 0.35
_HEURISTIC_SLOPE = 4.0


def heuristic_stop_probability(features: dict[str, float]) -> float:
    """Smooth, deterministic prior used before a trained model is available.

    Returns a value in (0, 1). The sigmoid crosses 0.5 at a weighted score of
    0.35, which corresponds roughly to a trajectory with several failures and
    repeated actions and no goal progress.
    """
    raw = sum(w * float(features.get(k, 0.0)) for k, w in HEURISTIC_WEIGHTS.items())
    z = _HEURISTIC_SLOPE * (raw - _HEURISTIC_PIVOT)
    return float(1.0 / (1.0 + math.exp(-z)))
