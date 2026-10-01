from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status

from convolve.db.models import Session as SessionRow
from convolve.db.models import Step as StepRow
from convolve.distiller.playbook import playbook_prompt_snippet
from convolve.proxy.deps import Container
from convolve.state.tracker import looks_like_error
from convolve.types import (
    CreateSessionRequest,
    CreateSessionResponse,
    Decision,
    EndSessionRequest,
    EndSessionResponse,
    HealthResponse,
    ObservationRequest,
    ObservationResponse,
    Outcome,
    PlaybookView,
    ProposeStepRequest,
    ProposeStepResponse,
    SessionStatus,
    SessionView,
)

log = logging.getLogger(__name__)
router = APIRouter()


def _container(request: Request) -> Container:
    return request.app.state.container


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/healthz", tags=["ops"])
async def healthz(request: Request) -> HealthResponse:
    c = _container(request)
    from convolve.version import __version__

    return HealthResponse(
        status="ok",
        version=__version__,
        state_backend=c.store.backend_name(),
        evaluator_source=c.evaluator.source,
    )


@router.get("/readyz", tags=["ops"])
async def readyz(request: Request) -> dict:
    c = _container(request)
    try:
        async with c.db.session() as s:
            from sqlalchemy import text

            await s.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"database not ready: {exc}") from exc


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@router.post(
    "/v1/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["sessions"],
)
async def create_session(
    request: Request, body: CreateSessionRequest
) -> CreateSessionResponse:
    c = _container(request)
    s = c.settings
    max_steps = body.max_steps or s.max_steps
    budget = body.budget if body.budget is not None else s.default_budget
    threshold = body.threshold if body.threshold is not None else s.default_threshold

    traj = await c.tracker.create(
        task=body.task, max_steps=max_steps, budget=budget, threshold=threshold
    )

    async with c.db.session() as db:
        db.add(
            SessionRow(
                id=traj.session_id,
                task=body.task,
                max_steps=max_steps,
                budget=budget,
                spent=0.0,
                threshold=threshold,
                status=SessionStatus.OPEN.value,
                meta=body.metadata or {},
                created_at=datetime.now(UTC),
            )
        )

    log.info("session created", extra={"session_id": traj.session_id})
    return CreateSessionResponse(
        session_id=traj.session_id,
        task=body.task,
        max_steps=max_steps,
        budget=budget,
        threshold=threshold,
        created_at=datetime.now(UTC).isoformat(),
    )


@router.get("/v1/sessions/{session_id}", response_model=SessionView, tags=["sessions"])
async def get_session(request: Request, session_id: str) -> SessionView:
    c = _container(request)
    traj = await c.tracker.load(session_id)
    if traj is None:
        raise HTTPException(status_code=404, detail="session not found")

    async with c.db.session() as db:
        row = await db.get(SessionRow, session_id)
        outcome = Outcome(row.outcome) if row and row.outcome else None
        status_val = SessionStatus(row.status) if row else SessionStatus.OPEN

    return SessionView(
        session_id=traj.session_id,
        task=traj.task,
        status=status_val,
        outcome=outcome,
        max_steps=traj.max_steps,
        budget=traj.budget,
        spent=traj.spent,
        threshold=traj.threshold,
        steps=[
            {
                "idx": s.idx,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation,
                "estimated_cost": s.estimated_cost,
                "actual_cost": s.actual_cost,
                "is_error": s.is_error,
                "stop_probability": s.stop_probability,
                "threshold_used": s.threshold_used,
                "decision": s.decision.value if s.decision else None,
                "reason": s.reason.value if s.reason else None,
            }
            for s in traj.steps
        ],
    )


# ---------------------------------------------------------------------------
# Step lifecycle
# ---------------------------------------------------------------------------


@router.post(
    "/v1/sessions/{session_id}/step",
    response_model=ProposeStepResponse,
    tags=["steps"],
)
async def propose_step(
    request: Request, session_id: str, body: ProposeStepRequest
) -> ProposeStepResponse:
    c = _container(request)
    traj = await c.tracker.load(session_id)
    if traj is None:
        raise HTTPException(status_code=404, detail="session not found")

    if c.tracker.pending_step(traj) is not None:
        raise HTTPException(
            status_code=409,
            detail="previous step has no observation yet; call /observation first",
        )

    verdict = c.evaluator.evaluate(traj)

    playbook_id: str | None = None
    if verdict.decision is Decision.ABSTAIN:
        best = await c.retriever.best(traj.task)
        if best is not None:
            playbook_id = best.playbook.id

    step = c.tracker.append_proposed(
        traj,
        thought=body.thought,
        action=body.action,
        action_input=body.action_input,
        estimated_cost=body.estimated_cost,
        stop_probability=verdict.stop_probability,
        threshold_used=verdict.threshold_used,
        base_threshold=verdict.base_threshold,
        reason=verdict.reason,
    )
    await c.tracker.save(traj)

    async with c.db.session() as db:
        db.add(
            StepRow(
                session_id=traj.session_id,
                idx=step.idx,
                thought=step.thought,
                action=step.action,
                action_input=step.action_input,
                estimated_cost=step.estimated_cost,
                stop_probability=step.stop_probability,
                threshold_used=step.threshold_used,
                decision=step.decision.value if step.decision else None,
                reason=step.reason.value if step.reason else None,
                created_at=datetime.now(UTC),
            )
        )

    log.info(
        "step proposed",
        extra={
            "session_id": traj.session_id,
            "step_index": step.idx,
            "decision": verdict.decision.value,
            "stop_probability": verdict.stop_probability,
        },
    )

    abstention_message: str | None = None
    if verdict.decision is Decision.ABSTAIN:
        abstention_message = c.evaluator.build_abstention_message(traj, verdict)
        if playbook_id is not None:
            pb = await c.playbook_store.get(playbook_id)
            if pb is not None:
                await c.playbook_store.record_usage(playbook_id)
                abstention_message = (
                    f"{abstention_message}\n\n{playbook_prompt_snippet(pb.rule_text)}"
                )

    return ProposeStepResponse(
        decision=verdict.decision,
        stop_probability=verdict.stop_probability,
        threshold=verdict.threshold_used,
        base_threshold=verdict.base_threshold,
        reason=verdict.reason,
        step_index=step.idx,
        playbook_id=playbook_id,
        abstention_message=abstention_message,
    )


@router.post(
    "/v1/sessions/{session_id}/observation",
    response_model=ObservationResponse,
    tags=["steps"],
)
async def report_observation(
    request: Request, session_id: str, body: ObservationRequest
) -> ObservationResponse:
    c = _container(request)
    traj = await c.tracker.load(session_id)
    if traj is None:
        raise HTTPException(status_code=404, detail="session not found")

    pending = c.tracker.pending_step(traj)
    if pending is None:
        raise HTTPException(
            status_code=409, detail="no pending step to attach observation to"
        )

    if pending.decision is not Decision.CONTINUE:
        raise HTTPException(
            status_code=409,
            detail="the last step was an abstention; no observation is expected",
        )

    is_error = (
        body.is_error if body.is_error is not None else looks_like_error(body.observation)
    )
    step = c.tracker.attach_observation(
        traj, observation=body.observation, cost=body.cost, is_error=is_error
    )
    await c.tracker.save(traj)

    async with c.db.session() as db:
        row = await db.get(SessionRow, session_id)
        if row is not None:
            row.spent = traj.spent

        from sqlalchemy import select

        srow = (
            await db.execute(
                select(StepRow).where(
                    StepRow.session_id == session_id, StepRow.idx == step.idx
                )
            )
        ).scalar_one_or_none()
        if srow is not None:
            srow.observation = step.observation
            srow.actual_cost = step.actual_cost
            srow.is_error = step.is_error

    return ObservationResponse(ok=True, step_index=step.idx, cumulative_cost=traj.spent)


@router.post(
    "/v1/sessions/{session_id}/end",
    response_model=EndSessionResponse,
    tags=["sessions"],
)
async def end_session(
    request: Request, session_id: str, body: EndSessionRequest
) -> EndSessionResponse:
    c = _container(request)
    traj = await c.tracker.load(session_id)
    if traj is None:
        raise HTTPException(status_code=404, detail="session not found")

    ended_at = datetime.now(UTC)

    async with c.db.session() as db:
        row = await db.get(SessionRow, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="session row not found")
        row.status = SessionStatus.ENDED.value
        row.outcome = body.outcome.value
        row.final_answer = body.final_answer
        row.spent = traj.spent
        row.ended_at = ended_at

    trajectory_path = _write_trajectory(c.settings.trajectory_dir, traj, body)
    await c.tracker.close(session_id)

    log.info(
        "session ended",
        extra={"session_id": session_id, "decision": body.outcome.value},
    )

    return EndSessionResponse(
        ok=True,
        session_id=session_id,
        outcome=body.outcome,
        total_steps=traj.n_steps,
        total_cost=traj.spent,
        trajectory_path=trajectory_path,
    )


def _write_trajectory(
    directory: Path, traj, body: EndSessionRequest
) -> str | None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trajectories.jsonl"
        record = {
            "session_id": traj.session_id,
            "task": traj.task,
            "max_steps": traj.max_steps,
            "budget": traj.budget,
            "threshold": traj.threshold,
            "spent": traj.spent,
            "outcome": body.outcome.value,
            "final_answer": body.final_answer,
            "ended_at": datetime.now(UTC).isoformat(),
            "steps": [
                {
                    "idx": s.idx,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "estimated_cost": s.estimated_cost,
                    "actual_cost": s.actual_cost,
                    "is_error": s.is_error,
                    "stop_probability": s.stop_probability,
                    "threshold_used": s.threshold_used,
                    "decision": s.decision.value if s.decision else None,
                    "reason": s.reason.value if s.reason else None,
                }
                for s in traj.steps
            ],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(path)
    except Exception as exc:  # pragma: no cover
        log.warning("failed to write trajectory: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Playbooks
# ---------------------------------------------------------------------------


@router.get("/v1/playbooks", response_model=list[PlaybookView], tags=["playbooks"])
async def list_playbooks(request: Request, limit: int = 200) -> list[PlaybookView]:
    c = _container(request)
    return await c.playbook_store.list_all(limit=limit)


@router.get("/v1/playbooks/{playbook_id}", response_model=PlaybookView, tags=["playbooks"])
async def get_playbook(request: Request, playbook_id: str) -> PlaybookView:
    c = _container(request)
    pb = await c.playbook_store.get(playbook_id)
    if pb is None:
        raise HTTPException(status_code=404, detail="playbook not found")
    return pb


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


@router.get("/v1/tasks/{session_id}/playbooks", tags=["playbooks"])
async def suggest_playbooks(request: Request, session_id: str) -> dict:
    c = _container(request)
    traj = await c.tracker.load(session_id)
    if traj is None:
        raise HTTPException(status_code=404, detail="session not found")

    results = await c.retriever.retrieve(traj.task)
    return {
        "task": traj.task,
        "suggestions": [
            {
                "playbook_id": r.playbook.id,
                "name": r.playbook.name,
                "rule_text": r.playbook.rule_text,
                "similarity": r.similarity,
                "prompt_snippet": playbook_prompt_snippet(r.playbook.rule_text),
            }
            for r in results
        ],
    }
