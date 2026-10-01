# Architecture

Convolve is a **gatekeeper proxy**. The client (an LLM agent) proposes an
action; Convolve either approves it or abstains. The proxy never executes the
tool itself, so no tool credentials are exposed to Convolve.

## Components

### 1. Interceptor proxy

FastAPI service exposing five routes under `/v1`. Each session is a state
machine with two legal transitions at any point in time:

    (open, no pending step)  --propose-->  (open, pending step)
    (open, pending step)     --observe-->  (open, no pending step)
    (open, any)              --end----->   (ended)

Any other transition returns HTTP 409.

### 2. State tracker

Hot state (the in-flight trajectory) lives in Redis under
`convolve:traj:{session_id}` with a TTL. When `CONVOLVE_REDIS_URL` is empty,
an in-process dictionary is used instead. Cold state (completed sessions,
steps, playbooks) lives in Postgres via SQLAlchemy async.

### 3. Abstention evaluator

Stateless. Given a trajectory it:

1. Extracts a fixed feature vector (`evaluator/features.py`).
2. Produces a raw stop-probability from either a trained
   `EvaluatorArtifact` (scikit-learn pipeline) or the deterministic
   heuristic prior.
3. Optionally applies isotonic calibration.

It then applies a three-tier decision:

    1. n_steps >= max_steps          => abstain (max_steps_reached)
    2. cost controller forces 0.0     => abstain (cost_budget_reached)
    3. stop_probability >= threshold  => abstain (uncertainty_threshold)
    4. otherwise                       => continue

### 4. Cost controller

Lowers the effective abstention threshold as the fraction of the budget spent
increases. The parameter `knee` (default 0.5) marks the point at which the
threshold begins to fall. At 100% of budget the threshold is 0.0 and the
decision is forced.

### 5. Trajectory distiller

Offline pipeline. Reads `data/trajectories/*.jsonl`, computes per-step
return-to-go from the session outcome, trains a Decision Transformer to
predict action tokens (including ABSTAIN), and mines the trained model for
playbooks.

### 6. Playbook store and retriever

`PlaybookStore` persists stopping rules in Postgres. `PlaybookRetriever` is a
lexical, intent-signature-keyed retriever that surfaces the top-k rules for a
task. When a session abstains, the top playbook's rule text is appended to the
abstention message so the agent gets a human-readable reason.

## Data flow

```
create -> tracker.create -> store.put(traj) -> db.insert(session)

step   -> tracker.load
       -> tracker.pending_step(traj) == None else 409
       -> evaluator.evaluate(traj)
       -> tracker.append_proposed(...)
       -> store.put(traj)
       -> db.insert(step)
       -> if abstain: retriever.best(task), db.update(playbook.usage_count)

observation
       -> tracker.load
       -> pending_step(traj) is not None else 409
       -> pending.decision == CONTINUE else 409
       -> tracker.attach_observation(...)
       -> store.put(traj); db.update(step, session.spent)

end    -> tracker.load
       -> db.update(session.status, outcome)
       -> write JSONL line to data/trajectories/trajectories.jsonl
       -> store.delete(session)
```

## Design decisions

| Decision | Rationale |
|---|---|
| Gatekeeper (not a transparent HTTP proxy) | Avoids storing tool credentials in Convolve; keeps the proxy framework-agnostic. |
| Feature-based evaluator (not an LLM-as-judge) | Low latency, deterministic, cheap to run per step. |
| Isotonic calibration | Non-parametric, monotone, works well on small datasets. |
| Decision Transformer with hashing tokenizer | Self-contained; no external tokenizer dependency; trains on CPU in minutes. |
| Redis hot / Postgres cold | Matches the access pattern; the trajectory is read and written every step. |
| JSONL trajectory log | Append-only, one line per session, easy to stream into training. |
