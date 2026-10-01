from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from convolve.distiller.rewards import RewardConfig, outcome_reward, return_to_go

PAD = 0
ABSTAIN = 1
BOS = 2
HASH_OFFSET = 3
VOCAB_SIZE = 1024

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, max_tokens: int = 64) -> list[int]:
    """Deterministic hashing tokenizer.

    We do not need a semantic tokenizer because the model conditions on
    return-to-go and the action/observation stream is small. A hashing
    tokenizer keeps the distiller self-contained and reproducible.
    """
    if not text:
        return []
    toks = _TOKEN_RE.findall(text.lower())[:max_tokens]
    out: list[int] = []
    for t in toks:
        h = hash(t) & 0x7FFFFFFF
        out.append(HASH_OFFSET + (h % (VOCAB_SIZE - HASH_OFFSET)))
    return out


@dataclass(slots=True)
class DistillerSample:
    """One trajectory flattened into a token sequence with RTG targets."""

    tokens: np.ndarray  # int64, shape (L,)
    rtg: np.ndarray  # float32, shape (L,)
    action_targets: np.ndarray  # int64, shape (L,) - token id at each position
    outcome: str
    session_id: str
    task: str
    abstain_index: int | None = None


def _step_text(step: dict) -> str:
    parts = [
        str(step.get("thought", "")),
        str(step.get("action", "")),
        json.dumps(step.get("action_input", {}), sort_keys=True, default=str),
        str(step.get("observation", "") or ""),
    ]
    return " ".join(p for p in parts if p)


def encode_trajectory(raw: dict, cfg: RewardConfig | None = None) -> DistillerSample:
    """Convert a raw trajectory JSON record into a training sample."""
    cfg = cfg or RewardConfig()
    steps = raw.get("steps", [])
    n = len(steps)
    term = outcome_reward(raw.get("outcome", "error"), cfg)
    rtg = return_to_go(n, term, cfg.step_cost_penalty)

    tokens: list[int] = [BOS]
    rtg_list: list[float] = [float(term)]
    targets: list[int] = [BOS]
    abstain_index: int | None = None

    for i, s in enumerate(steps):
        toks = tokenize(_step_text(s))
        if not toks:
            toks = [BOS]
        tokens.extend(toks)
        rtg_list.extend([float(rtg[i])] * len(toks))
        # Predict the first token of each step from the preceding context.
        targets.extend(toks)
        if s.get("decision") == "abstain" and abstain_index is None:
            abstain_index = len(tokens) - len(toks)

    return DistillerSample(
        tokens=np.asarray(tokens, dtype=np.int64),
        rtg=np.asarray(rtg_list, dtype=np.float32),
        action_targets=np.asarray(targets, dtype=np.int64),
        outcome=raw.get("outcome", "error"),
        session_id=raw.get("session_id", ""),
        task=raw.get("task", ""),
        abstain_index=abstain_index,
    )


def load_trajectories(path: Path) -> list[dict]:
    records: list[dict] = []
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path]
    for f in files:
        if not f.exists():
            continue
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def iter_samples(
    records: Iterable[dict], cfg: RewardConfig | None = None
) -> Iterator[DistillerSample]:
    for r in records:
        if not r.get("steps"):
            continue
        yield encode_trajectory(r, cfg)


@dataclass
class TrajectoryDataset:
    """Padded tensors ready for the Decision Transformer."""

    tokens: np.ndarray  # (N, L) int64
    rtg: np.ndarray  # (N, L) float32
    targets: np.ndarray  # (N, L) int64
    mask: np.ndarray  # (N, L) float32 - 1 for real tokens
    outcomes: list[str]
    tasks: list[str]
    session_ids: list[str]
    abstain_indices: list[int | None]
    length: int

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def train_val_split(self, val_frac: float = 0.2, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = len(self)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_frac)) if n > 1 else 0
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return self._subset(train_idx), self._subset(val_idx)

    def _subset(self, indices: np.ndarray) -> TrajectoryDataset:
        return TrajectoryDataset(
            tokens=self.tokens[indices],
            rtg=self.rtg[indices],
            targets=self.targets[indices],
            mask=self.mask[indices],
            outcomes=[self.outcomes[i] for i in indices],
            tasks=[self.tasks[i] for i in indices],
            session_ids=[self.session_ids[i] for i in indices],
            abstain_indices=[self.abstain_indices[i] for i in indices],
            length=self.length,
        )


def build_dataset(
    records: list[dict],
    max_len: int = 128,
    cfg: RewardConfig | None = None,
) -> TrajectoryDataset:
    samples = list(iter_samples(records, cfg))
    if not samples:
        raise ValueError("no usable trajectories found")

    N = len(samples)
    L = max_len

    tokens = np.zeros((N, L), dtype=np.int64)
    rtg = np.zeros((N, L), dtype=np.float32)
    targets = np.zeros((N, L), dtype=np.int64)
    mask = np.zeros((N, L), dtype=np.float32)

    for i, s in enumerate(samples):
        take = min(L, s.tokens.shape[0])
        tokens[i, :take] = s.tokens[:take]
        rtg[i, :take] = s.rtg[:take]
        targets[i, :take] = s.action_targets[:take]
        mask[i, :take] = 1.0

    return TrajectoryDataset(
        tokens=tokens,
        rtg=rtg,
        targets=targets,
        mask=mask,
        outcomes=[s.outcome for s in samples],
        tasks=[s.task for s in samples],
        session_ids=[s.session_id for s in samples],
        abstain_indices=[s.abstain_index for s in samples],
        length=L,
    )
