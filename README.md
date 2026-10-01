# Convolve

**Agentic abstention and sequential uncertainty middleware for LLM agents.**

Convolve sits between an LLM agent and its tools. It watches the full
interaction trajectory, scores how likely the task is to be impossible, and
forces the agent to stop when that score crosses a threshold. It learns
reusable stopping rules ("playbooks") from past trajectories via a Decision
Transformer, and it can halt an agent purely on cost grounds when a budget is
about to be exhausted.

## Why

LLM agents loop on impossible tasks. A 2025 study found that Llama-3.3-70B and
Gemini loop on 30–60% of impossible tasks in a standardised benchmark. Existing
guards (`max_iterations`, output validators) evaluate one step at a time and
never learn from past failures. Convolve evaluates the *whole trajectory* and
improves over time.

## Install

```bash
git clone <repo> convolve && cd convolve
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

For the Decision Transformer (offline training only):

```bash
pip install -e ".[torch]"
```

## Run locally (SQLite + in-memory state, no Docker)

```bash
export CONVOLVE_DATABASE_URL="sqlite+aiosqlite:///./convolve.db"
export CONVOLVE_REDIS_URL=""          # empty => in-process state store
convolve init-db
convolve serve
```

The API is at `http://localhost:8000`. Open `http://localhost:8000/ui` for the
trajectory inspector.

## Run with Docker

```bash
docker compose up --build
```

Starts Postgres, Redis, and the Convolve proxy.

## Minimal usage

```python
import httpx

with httpx.Client(base_url="http://localhost:8000") as c:
    sid = c.post("/v1/sessions", json={
        "task": "Book a flight BOM->DEL under Rs.500",
        "max_steps": 12,
        "budget": 0.50,
    }).json()["session_id"]

    while True:
        d = c.post(f"/v1/sessions/{sid}/step", json={
            "thought": "Search for flights",
            "action": "search_flights",
            "action_input": {"origin": "BOM", "dest": "DEL", "max_price": 500},
            "estimated_cost": 0.05,
        }).json()

        if d["decision"] == "abstain":
            print("Stopped:", d["abstention_message"])
            break

        obs = run_my_tool(d["action"], d["action_input"])   # your code
        c.post(f"/v1/sessions/{sid}/observation", json={
            "observation": obs, "cost": 0.05, "is_error": False,
        })

    c.post(f"/v1/sessions/{sid}/end", json={"outcome": "abstained"})
```

See `examples/` for a LangChain callback handler.

## CLI

| Command | Purpose |
|---|---|
| `convolve serve` | Run the FastAPI proxy |
| `convolve init-db` | Apply Alembic migrations |
| `convolve simulate` | Write synthetic trajectories to `data/trajectories/` |
| `convolve train-evaluator` | Fit + calibrate the abstention classifier |
| `convolve train-distiller` | Train the Decision Transformer |
| `convolve extract-playbooks` | Mine playbooks from the trained distiller |
| `convolve eval` | Evaluate the evaluator on a labelled trajectory set |

## Tests

```bash
make test
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/api.md`](docs/api.md)
- [`docs/integration.md`](docs/integration.md)
- [`docs/training.md`](docs/training.md)
- [`docs/deployment.md`](docs/deployment.md)

## License

Apache-2.0. See `LICENSE`.
