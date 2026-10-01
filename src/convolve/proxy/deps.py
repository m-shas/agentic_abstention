from __future__ import annotations

from dataclasses import dataclass

from convolve.config import Settings
from convolve.cost.controller import CostController
from convolve.db.session import Database
from convolve.evaluator.evaluator import AbstentionEvaluator
from convolve.playbooks.retrieval import PlaybookRetriever
from convolve.playbooks.store import PlaybookStore
from convolve.state.store import InMemoryStateStore, RedisStateStore, StateStore
from convolve.state.tracker import StateTracker


@dataclass
class Container:
    """Application-wide singletons, attached to `app.state.container`."""

    settings: Settings
    db: Database
    store: StateStore
    tracker: StateTracker
    cost: CostController
    evaluator: AbstentionEvaluator
    playbook_store: PlaybookStore
    retriever: PlaybookRetriever

    async def aclose(self) -> None:
        await self.store.close()


def build_container(settings: Settings, db: Database) -> Container:
    settings.ensure_dirs()

    if settings.use_redis:
        store: StateStore = RedisStateStore(
            settings.redis_url, ttl_seconds=settings.state_ttl_seconds
        )
    else:
        store = InMemoryStateStore(ttl_seconds=settings.state_ttl_seconds)

    tracker = StateTracker(store)
    cost = CostController(knee=settings.cost_knee)
    evaluator = AbstentionEvaluator(settings, cost)
    playbook_store = PlaybookStore(db)
    retriever = PlaybookRetriever(playbook_store, top_k=settings.playbook_top_k)

    return Container(
        settings=settings,
        db=db,
        store=store,
        tracker=tracker,
        cost=cost,
        evaluator=evaluator,
        playbook_store=playbook_store,
        retriever=retriever,
    )
