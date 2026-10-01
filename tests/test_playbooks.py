from __future__ import annotations

import pytest

from convolve.db.session import Database
from convolve.playbooks.retrieval import PlaybookRetriever, task_signature
from convolve.playbooks.store import PlaybookStore


@pytest.fixture
async def store(settings):
    db = Database(settings.database_url)
    await db.create_all()
    try:
        yield PlaybookStore(db)
    finally:
        await db.drop_all()
        await db.dispose()


def test_signature_stable_and_distinct():
    a = task_signature("Book a flight from BOM to DEL under Rs.500")
    b = task_signature("Book a flight from BOM to DEL on 2026-08-15")
    c = task_signature("Validate coupon FREE100")
    assert a == b  # both start with "book"
    assert a != c
    # Deterministic
    assert a == task_signature("Book a flight from BOM to DEL under Rs.500")


async def test_upsert_and_list(store: PlaybookStore):
    pb = await store.upsert(
        name="pb-1",
        task_signature="book:abcd1234",
        task_exemplar="Book a cheap flight",
        rule_text="Stop after 3 search_flights failures.",
        abstain_at_step=3,
        source_session_id=None,
        score=0.9,
    )
    assert pb.id
    all_pbs = await store.list_all()
    assert len(all_pbs) == 1
    assert all_pbs[0].rule_text == "Stop after 3 search_flights failures."


async def test_get_and_usage(store: PlaybookStore):
    pb = await store.upsert(
        name="pb-2",
        task_signature="book:00001111",
        task_exemplar="Book",
        rule_text="Rule",
        abstain_at_step=2,
        source_session_id=None,
        score=0.5,
    )
    got = await store.get(pb.id)
    assert got is not None
    assert got.usage_count == 0
    await store.record_usage(pb.id)
    got = await store.get(pb.id)
    assert got.usage_count == 1


async def test_count_and_clear(store: PlaybookStore):
    for i in range(3):
        await store.upsert(
            name=f"pb-{i}",
            task_signature=f"sig:{i}",
            task_exemplar="x",
            rule_text="y",
            abstain_at_step=1,
            source_session_id=None,
            score=0.1 * i,
        )
    assert await store.count() == 3
    await store.clear()
    assert await store.count() == 0


async def test_retriever_ranks_by_similarity(store: PlaybookStore):
    await store.upsert(
        name="flight-cheap",
        task_signature=task_signature("Book a flight under Rs.500"),
        task_exemplar="Book a flight under Rs.500 from BOM to DEL",
        rule_text="Stop after 2 search_flights failures.",
        abstain_at_step=2,
        source_session_id=None,
        score=0.9,
    )
    await store.upsert(
        name="coupon",
        task_signature=task_signature("Validate a coupon code"),
        task_exemplar="Validate coupon code FREE100",
        rule_text="Stop after 3 validate_coupon failures.",
        abstain_at_step=3,
        source_session_id=None,
        score=0.9,
    )
    retriever = PlaybookRetriever(store, top_k=2)
    results = await retriever.retrieve("Book a flight under Rs.500 from BOM to DEL")
    assert results
    assert results[0].playbook.name == "flight-cheap"


async def test_retriever_zero_topk(store: PlaybookStore):
    retriever = PlaybookRetriever(store, top_k=0)
    assert await retriever.retrieve("anything") == []
