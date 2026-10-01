"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-01

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("max_steps", sa.Integer(), nullable=False, server_default="12"),
        sa.Column("budget", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("spent", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("threshold", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("outcome", sa.String(16), nullable=True),
        sa.Column("final_answer", sa.Text(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sessions_status", "sessions", ["status"])
    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])

    op.create_table(
        "steps",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.String(64),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("thought", sa.Text(), nullable=False, server_default=""),
        sa.Column("action", sa.String(128), nullable=False, server_default=""),
        sa.Column("action_input", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("observation", sa.Text(), nullable=True),
        sa.Column("estimated_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("actual_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("is_error", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("stop_probability", sa.Float(), nullable=True),
        sa.Column("threshold_used", sa.Float(), nullable=True),
        sa.Column("decision", sa.String(16), nullable=True),
        sa.Column("reason", sa.String(48), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_steps_session_idx", "steps", ["session_id", "idx"], unique=True)

    op.create_table(
        "playbooks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("task_signature", sa.String(64), nullable=False),
        sa.Column("task_exemplar", sa.Text(), nullable=False, server_default=""),
        sa.Column("rule_text", sa.Text(), nullable=False),
        sa.Column("abstain_at_step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_session_id", sa.String(64), nullable=True),
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_playbooks_signature", "playbooks", ["task_signature"])
    op.create_index("ix_playbooks_score", "playbooks", ["score"])


def downgrade() -> None:
    op.drop_index("ix_playbooks_score", table_name="playbooks")
    op.drop_index("ix_playbooks_signature", table_name="playbooks")
    op.drop_table("playbooks")
    op.drop_index("ix_steps_session_idx", table_name="steps")
    op.drop_table("steps")
    op.drop_index("ix_sessions_created_at", table_name="sessions")
    op.drop_index("ix_sessions_status", table_name="sessions")
    op.drop_table("sessions")
