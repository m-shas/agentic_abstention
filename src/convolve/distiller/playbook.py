from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from convolve.distiller.dataset import load_trajectories, tokenize
from convolve.distiller.decision_transformer import DTConfig, is_available
from convolve.playbooks.retrieval import task_signature
from convolve.playbooks.store import PlaybookStore

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class ExtractedPlaybook:
    name: str
    task_signature: str
    task_exemplar: str
    rule_text: str
    abstain_at_step: int
    source_session_id: str | None
    score: float


def _shorten(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _build_rule_text(task: str, abstain_at_step: int, n_steps: int, outcomes: str) -> str:
    return (
        f"When a task resembling '{_shorten(task)}' has produced {abstain_at_step} "
        f"steps without resolving (outcome: {outcomes}), abstain instead of "
        f"continuing past step {abstain_at_step}. Observed full trajectory length "
        f"was {n_steps} steps."
    )


def _model_abstain_step(
    model, cfg: DTConfig, record: dict, threshold: float = 0.5
) -> int | None:
    """Ask the trained DT at which step it would emit ABSTAIN.

    Returns the step index, or None if the model never crosses `threshold`.
    """
    import torch

    steps = record.get("steps", [])
    if not steps:
        return None

    tokens: list[int] = [2]  # BOS
    step_offsets: list[int] = [0]

    for s in steps:
        text = " ".join(
            [
                str(s.get("thought", "")),
                str(s.get("action", "")),
                str(s.get("observation", "") or ""),
            ]
        )
        toks = tokenize(text) or [2]
        step_offsets.append(len(tokens) + len(toks))
        tokens.extend(toks)

    L = min(len(tokens), cfg.max_len)
    toks = torch.tensor([tokens[:L]], dtype=torch.long)
    rtg = torch.full((1, L), 1.0, dtype=torch.float32)
    mask = torch.ones((1, L), dtype=torch.float32)

    with torch.no_grad():
        probs = model.score_abstain(toks, rtg, mask)[0].cpu().numpy()

    for i, offset in enumerate(step_offsets[:-1]):
        if offset >= L:
            break
        if probs[offset] >= threshold:
            return i

    return None


def extract_playbooks(
    trajectories_path: Path,
    checkpoint_path: Path,
    store: PlaybookStore,
    *,
    max_playbooks: int = 100,
    abstain_threshold: float = 0.5,
    only_outcomes: tuple[str, ...] = ("looped", "error", "aborted"),
) -> list[ExtractedPlaybook]:
    """Mine stopping rules from failing trajectories using the trained DT.

    For every failing trajectory the DT is asked where it would have abstained.
    The resulting (task, step, outcome) tuple becomes a playbook row.
    """
    if not is_available():
        raise RuntimeError("PyTorch is required for playbook extraction")

    from convolve.distiller.train import load_model

    model, cfg, *_meta = load_model(checkpoint_path)

    records = load_trajectories(trajectories_path)
    candidates = [r for r in records if r.get("outcome") in only_outcomes]

    if not candidates:
        log.warning("no failing trajectories found for playbook extraction")
        return []

    extracted: list[ExtractedPlaybook] = []
    seen_signatures: set[tuple[str, int]] = set()

    for rec in candidates:
        if len(extracted) >= max_playbooks:
            break

        step = _model_abstain_step(model, cfg, rec, threshold=abstain_threshold)

        if step is None:
            # Fall back to the first error step, if any.
            for i, s in enumerate(rec.get("steps", [])):
                if s.get("is_error"):
                    step = i
                    break

        if step is None or step == 0:
            continue

        sig = task_signature(rec.get("task", ""))
        key = (sig, step)
        if key in seen_signatures:
            continue
        seen_signatures.add(key)

        outcome = rec.get("outcome", "error")
        n_steps = len(rec.get("steps", []))
        score = 0.6 + 0.4 * min(1.0, step / max(1, n_steps))

        extracted.append(
            ExtractedPlaybook(
                name=f"pb-{sig}-{step}",
                task_signature=sig,
                task_exemplar=rec.get("task", ""),
                rule_text=_build_rule_text(rec.get("task", ""), step, n_steps, outcome),
                abstain_at_step=step,
                source_session_id=rec.get("session_id"),
                score=round(score, 4),
            )
        )

    log.info("extracted %d playbooks", len(extracted))
    return extracted


async def persist_playbooks(store: PlaybookStore, playbooks: list[ExtractedPlaybook]) -> int:
    count = 0
    for pb in playbooks:
        await store.upsert(
            name=pb.name,
            task_signature=pb.task_signature,
            task_exemplar=pb.task_exemplar,
            rule_text=pb.rule_text,
            abstain_at_step=pb.abstain_at_step,
            source_session_id=pb.source_session_id,
            score=pb.score,
        )
        count += 1
    return count


def playbook_prompt_snippet(rule_text: str) -> str:
    """Format a rule for injection into an agent's system prompt."""
    return (
        "STOPPING RULE (learned from past experience):\n"
        f"- {rule_text}\n"
        "If the current situation matches this rule, stop and explain why "
        "instead of continuing to call tools."
    )


__all__ = [
    "ExtractedPlaybook",
    "extract_playbooks",
    "persist_playbooks",
    "playbook_prompt_snippet",
]
