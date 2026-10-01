# Integration Guide

Convolve is a **gatekeeper proxy**, not a transparent HTTP proxy. To integrate
it you route every tool invocation through two calls: `POST /step` before
executing the tool, and `POST /observation` after.

## The five-call loop

```python
sid = POST /v1/sessions        # once per task

loop:
    d = POST /v1/sessions/{sid}/step
    if d.decision == "abstain":
        return d.abstention_message
    observation = run_tool(d.action, d.action_input)
    POST /v1/sessions/{sid}/observation

POST /v1/sessions/{sid}/end     # once per task
```

## LangChain

Register a callback handler on your `AgentExecutor`:

```python
from langchain_core.callbacks import BaseCallbackHandler

class ConvolveCallback(BaseCallbackHandler):
    def __init__(self, session: ConvolveSession):
        self._session = session

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._session.before_tool(
            thought="(from agent)",
            action=serialized["name"],
            action_input={"input": input_str},
        )

    def on_tool_end(self, output, **kwargs):
        self._session.after_tool(observation=str(output))

    def on_tool_error(self, error, **kwargs):
        self._session.after_tool(observation=str(error), is_error=True)
```

Wrap the executor call in a `try/except ConvolveAbstained` and return the
abstention message as the agent's final answer. See
`examples/langchain_demo.py` for a complete sketch.

## LlamaIndex

LlamaIndex exposes tools as `FunctionTool`. Wrap each tool function:

```python
def wrapped(fn, session):
    def _inner(**kwargs):
        session.before_tool(thought="", action=fn.__name__, action_input=kwargs)
        try:
            out = fn(**kwargs)
        except Exception as exc:
            session.after_tool(observation=str(exc), is_error=True)
            raise
        session.after_tool(observation=str(out))
        return out
    return _inner
```

## Raw HTTP / any framework

Use `httpx` (sync or async) as shown in `examples/http_client_demo.py`.

## Choosing thresholds

| Situation | Suggested threshold |
|---|---|
| Safety-critical (never let an agent loop) | `0.4` |
| Balanced default | `0.7` |
| Throughput-sensitive (avoid stopping valid tasks) | `0.9` |

## Choosing budgets

Set `budget` to the maximum API cost you are willing to spend on a single
task. The cost controller begins lowering the threshold at `knee × budget`
(default 50%) and forces abstention at `budget`.

## Injecting playbooks

Two options:

1. **Read the abstention message.** It already contains the top playbook's
   rule text when one applies.

2. **Pre-fetch suggestions.** Call `GET /v1/tasks/{sid}/playbooks` before the
   loop starts and prepend the `prompt_snippet` of the top result to your
   agent's system prompt. This nudges the agent to abstain on its own.

## Idempotency and retries

`POST /step` is **not** idempotent by design — it records a proposal. If the
network fails mid-request, check `GET /v1/sessions/{sid}` before retrying; if a
pending step exists, call `/observation` or `/end` instead of re-proposing.
