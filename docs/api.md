# HTTP API

Base URL: `http://localhost:8000`

All request and response bodies are JSON. All timestamps are ISO-8601 UTC.

## `GET /healthz`

Liveness probe.

**Response** `200`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "state_backend": "redis",
  "evaluator_source": "heuristic"
}
```

## `GET /readyz`

Readiness probe. Returns `503` if the database is unreachable.

## `POST /v1/sessions`

Create a new session.

**Body**

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `task` | string | yes | — | 1–8000 chars |
| `max_steps` | int | no | `CONVOLVE_MAX_STEPS` | 1–200 |
| `budget` | float | no | `CONVOLVE_DEFAULT_BUDGET` | >= 0 |
| `threshold` | float | no | `CONVOLVE_DEFAULT_THRESHOLD` | 0–1 |
| `metadata` | object | no | `{}` | Opaque; stored on the session |

**Response** `201`

```json
{
  "session_id": "8d3f...",
  "task": "Book a flight BOM->DEL under Rs.500",
  "max_steps": 12,
  "budget": 0.5,
  "threshold": 0.7,
  "created_at": "2026-08-01T09:00:00+00:00"
}
```

## `GET /v1/sessions/{session_id}`

Fetch the session including all recorded steps.

**Response** `200`

```json
{
  "session_id": "8d3f...",
  "task": "...",
  "status": "open",
  "outcome": null,
  "max_steps": 12,
  "budget": 0.5,
  "spent": 0.15,
  "threshold": 0.7,
  "steps": [
    {
      "idx": 0,
      "thought": "Search for flights",
      "action": "search_flights",
      "action_input": {"origin": "BOM", "dest": "DEL"},
      "observation": "no results found",
      "estimated_cost": 0.05,
      "actual_cost": 0.05,
      "is_error": true,
      "stop_probability": 0.28,
      "threshold_used": 0.7,
      "decision": "continue",
      "reason": "under_threshold"
    }
  ]
}
```

`404` if the session is not found.

## `POST /v1/sessions/{session_id}/step`

Propose an action. This is the abstention decision point.

**Body**

| Field | Type | Required |
|---|---|---|
| `thought` | string | no |
| `action` | string | yes (1–128 chars) |
| `action_input` | object | no |
| `estimated_cost` | float | no |

**Response** `200`

```json
{
  "decision": "continue",
  "stop_probability": 0.28,
  "threshold": 0.70,
  "base_threshold": 0.70,
  "reason": "under_threshold",
  "step_index": 0,
  "playbook_id": null,
  "abstention_message": null
}
```

Possible `reason` values:

* `under_threshold` — continue.
* `uncertainty_threshold` — abstain; the score crossed the threshold.
* `cost_budget_reached` — abstain; the budget is exhausted.
* `max_steps_reached` — abstain; the step budget is exhausted.

When `decision == "abstain"`, `abstention_message` is populated and
`playbook_id` may be set.

**Errors**

* `404` — session not found.
* `409` — a previous step has no observation yet.

## `POST /v1/sessions/{session_id}/observation`

Report the outcome of the approved action.

**Body**

| Field | Type | Required | Notes |
|---|---|---|---|
| `observation` | string | no | 0–100 000 chars |
| `cost` | float | no | Added to cumulative spend |
| `is_error` | bool | no | Auto-detected from `observation` if omitted |

**Response** `200`

```json
{"ok": true, "step_index": 0, "cumulative_cost": 0.05}
```

**Errors**

* `409` — no pending step, or the pending step was an abstention.

## `POST /v1/sessions/{session_id}/end`

Finalise the session.

**Body**

| Field | Type | Required |
|---|---|---|
| `outcome` | enum | yes |
| `final_answer` | string | no |

`outcome` is one of `success`, `abstained`, `looped`, `error`, `aborted`.

**Response** `200`

```json
{
  "ok": true,
  "session_id": "8d3f...",
  "outcome": "abstained",
  "total_steps": 3,
  "total_cost": 0.15,
  "trajectory_path": "data/trajectories/trajectories.jsonl"
}
```

## `GET /v1/playbooks`

List stored playbooks, highest score first.

## `GET /v1/playbooks/{playbook_id}`

Fetch a single playbook.

## `GET /v1/tasks/{session_id}/playbooks`

Suggest the top-k playbooks for a session's task.

```json
{
  "task": "Book a flight BOM->DEL under Rs.500",
  "suggestions": [
    {
      "playbook_id": "abc123",
      "name": "pb-book:abcd1234-3",
      "rule_text": "When a task resembling 'Book a flight...' has produced 3 steps without resolving ... abstain instead of continuing past step 3.",
      "similarity": 0.83,
      "prompt_snippet": "STOPPING RULE (learned from past experience):\n- ...\nIf the current situation matches this rule, stop and explain why instead of continuing to call tools."
    }
  ]
}
```

## Errors

All error responses use FastAPI's default envelope:

```json
{"detail": "human-readable message"}
```

Codes used: `404`, `409`, `422`, `500`, `503`.
