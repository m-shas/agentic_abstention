from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update

from convolve.db.models import Playbook as PlaybookRow
from convolve.types import PlaybookView


class PlaybookStore:
    """Persistence for distilled stopping rules."""

    def __init__(self, db) -> None:  # db: Database
        self._db = db

    async def upsert(
        self,
        *,
        name: str,
        task_signature: str,
        task_exemplar: str,
        rule_text: str,
        abstain_at_step: int,
        source_session_id: str | None,
        score: float,
    ) -> PlaybookView:
        pid = uuid.uuid4().hex
        async with self._db.session() as s:
            row = PlaybookRow(
                id=pid,
                name=name,
                task_signature=task_signature,
                task_exemplar=task_exemplar,
                rule_text=rule_text,
                abstain_at_step=abstain_at_step,
                source_session_id=source_session_id,
                score=score,
                usage_count=0,
                created_at=datetime.now(UTC),
            )
            s.add(row)
        return PlaybookView(
            id=pid,
            name=name,
            task_signature=task_signature,
            task_exemplar=task_exemplar,
            rule_text=rule_text,
            abstain_at_step=abstain_at_step,
            score=score,
            usage_count=0,
        )

    async def list_all(self, limit: int = 500) -> list[PlaybookView]:
        async with self._db.session() as s:
            rows = (
                await s.execute(
                    select(PlaybookRow).order_by(PlaybookRow.score.desc()).limit(limit)
                )
            ).scalars().all()
        return [self._to_view(r) for r in rows]

    async def get(self, playbook_id: str) -> PlaybookView | None:
        async with self._db.session() as s:
            row = await s.get(PlaybookRow, playbook_id)
        return self._to_view(row) if row else None

    async def record_usage(self, playbook_id: str) -> None:
        async with self._db.session() as s:
            await s.execute(
                update(PlaybookRow)
                .where(PlaybookRow.id == playbook_id)
                .values(usage_count=PlaybookRow.usage_count + 1)
            )

    async def count(self) -> int:
        from sqlalchemy import func

        async with self._db.session() as s:
            return int((await s.execute(select(func.count(PlaybookRow.id)))).scalar_one())

    async def clear(self) -> None:
        from sqlalchemy import delete

        async with self._db.session() as s:
            await s.execute(delete(PlaybookRow))

    @staticmethod
    def _to_view(row: PlaybookRow) -> PlaybookView:
        return PlaybookView(
            id=row.id,
            name=row.name,
            task_signature=row.task_signature,
            task_exemplar=row.task_exemplar,
            rule_text=row.rule_text,
            abstain_at_step=row.abstain_at_step,
            score=row.score,
            usage_count=row.usage_count,
        )
