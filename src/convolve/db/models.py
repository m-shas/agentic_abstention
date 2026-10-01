from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=12)
    budget: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.7)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict] = mapped_column("meta", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list[Step]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Step.idx",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_sessions_status", "status"),
        Index("ix_sessions_created_at", "created_at"),
    )


class Step(Base):
    __tablename__ = "steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    thought: Mapped[str] = mapped_column(Text, nullable=False, default="")
    action: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    action_input: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    observation: Mapped[str | None] = mapped_column(Text, nullable=True)
    estimated_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    actual_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    stop_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(48), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    session: Mapped[Session] = relationship(back_populates="steps")

    __table_args__ = (Index("ix_steps_session_idx", "session_id", "idx", unique=True),)


class Playbook(Base):
    __tablename__ = "playbooks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    task_signature: Mapped[str] = mapped_column(String(64), nullable=False)
    task_exemplar: Mapped[str] = mapped_column(Text, nullable=False, default="")
    rule_text: Mapped[str] = mapped_column(Text, nullable=False)
    abstain_at_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("ix_playbooks_signature", "task_signature"),
        Index("ix_playbooks_score", "score"),
    )
