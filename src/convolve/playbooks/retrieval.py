from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from convolve.playbooks.store import PlaybookStore
from convolve.types import PlaybookView

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """a an the is are was were be been to of in on for with and or not do does
    i you we they it this that my your our their at by from as if then than
    please can could would should""".split()
)


def task_signature(task: str) -> str:
    """Coarse signature of a task, used as a fast playbook index.

    We bucket by a small set of intent verbs so that, e.g., all "book ... under
    X" tasks collide and share playbooks.
    """
    tokens = [t for t in _TOKEN_RE.findall(task.lower()) if t not in _STOP]
    intent = tokens[0] if tokens else "unknown"
    bucket = hashlib.sha1(intent.encode()).hexdigest()[:8]
    return f"{intent}:{bucket}"


def _task_tokens(task: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(task.lower()) if t not in _STOP}


@dataclass(slots=True)
class RetrievedPlaybook:
    playbook: PlaybookView
    similarity: float


class PlaybookRetriever:
    """Lexical retriever over the playbook store.

    Deliberately simple: playbooks are keyed by intent signature, then ranked
    by Jaccard similarity on task tokens and by their stored score. This avoids
    a vector database dependency while still surfacing useful rules.
    """

    def __init__(self, store: PlaybookStore, top_k: int = 2) -> None:
        self._store = store
        self._top_k = top_k

    async def retrieve(self, task: str) -> list[RetrievedPlaybook]:
        if self._top_k <= 0:
            return []

        sig = task_signature(task)
        candidates = await self._store.list_all(limit=1000)
        q = _task_tokens(task)

        scored: list[RetrievedPlaybook] = []
        for pb in candidates:
            c = _task_tokens(pb.task_exemplar) | {pb.task_signature.split(":")[0]}
            if not c and not q:
                sim = 0.0
            else:
                sim = len(q & c) / max(1, len(q | c))
            if pb.task_signature == sig:
                sim += 0.35
            sim += 0.15 * pb.score
            scored.append(RetrievedPlaybook(playbook=pb, similarity=round(sim, 6)))

        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[: self._top_k]

    async def best(self, task: str) -> RetrievedPlaybook | None:
        results = await self.retrieve(task)
        return results[0] if results else None
