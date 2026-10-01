# Deployment Guide

## Docker Compose (single host)

```bash
docker compose up --build -d
```

Services:

| Service | Port | Notes |
|---|---|---|
| `convolve` | 8000 | The proxy; runs `convolve init-db && convolve serve` |
| `postgres` | 5432 | Cold storage |
| `redis` | 6379 | Hot state |

The `convolve` container mounts `./artifacts` and `./data` so that trained
models and trajectory logs survive container restarts.

### Zero-downtime redeploy

```bash
docker compose build convolve
docker compose up -d --no-deps --scale convolve=2 convolve
docker compose up -d --no-deps --scale convolve=1 convolve
```

## Kubernetes sketch

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: convolve
spec:
  replicas: 3
  selector:
    matchLabels: { app: convolve }
  template:
    metadata:
      labels: { app: convolve }
    spec:
      containers:
        - name: convolve
          image: your-registry/convolve:0.1.0
          ports: [{ containerPort: 8000 }]
          envFrom:
            - secretRef: { name: convolve-secrets }
          readinessProbe:
            httpGet: { path: /readyz, port: 8000 }
            initialDelaySeconds: 5
          livenessProbe:
            httpGet: { path: /healthz, port: 8000 }
            initialDelaySeconds: 10
          resources:
            requests: { cpu: "250m", memory: "256Mi" }
            limits:   { cpu: "1",    memory: "1Gi" }
---
apiVersion: v1
kind: Service
metadata: { name: convolve }
spec:
  selector: { app: convolve }
  ports: [{ port: 80, targetPort: 8000 }]
```

Store secrets in `convolve-secrets`:

```
CONVOLVE_DATABASE_URL=postgresql+asyncpg://...
CONVOLVE_REDIS_URL=redis://...
```

## Scaling notes

* The proxy is **stateless except for Redis**. Scale horizontally by adding
  replicas; all replicas share the same Redis and Postgres.
* Set `CONVOLVE_STATE_TTL_SECONDS` to at least `max_session_duration + 300s`.
* Postgres connection pool size defaults to SQLAlchemy's; for high
  throughput, set `pool_size` and `max_overflow` in `Database.__init__`.
* The evaluator artefact is loaded once per process. When retraining, roll
  out new pods rather than mutating a running process.

## Observability

* **Logs.** Structured JSON when `CONVOLVE_ENV=production`. Each step logs
  `session_id`, `step_index`, `decision`, and `stop_probability`.
* **Metrics.** Scrape `/healthz` and `/readyz`. For Prometheus metrics, mount
  `prometheus_fastapi_instrumentator` in `app.py` (not included by default to
  keep the dependency tree small).
* **Traces.** The proxy is a plain HTTP service; instrument it with OpenTelemetry
  via the standard FastAPI middleware.

## Backups

* Postgres: back up `sessions`, `steps`, `playbooks`. These are the source of
  truth for offline training and audit.
* Redis: no backup needed; the TTL means lost sessions are simply dropped.
* Trajectory JSONL under `data/trajectories/`: append-only, cheap to back up,
  and the primary input to retraining.

## Cost controls

* Set `CONVOLVE_DEFAULT_BUDGET` to your per-task ceiling.
* Set `CONVOLVE_COST_KNEE` to the fraction of budget at which the threshold
  begins to fall. `0.5` is a good default; `0.8` is safer for tasks where
  early mistakes are cheap.
* Monitor the ratio of `reason=cost_budget_reached` abstentions to total
  abstentions. A sudden spike means either the budget is too low or the agent
  is wasting calls.
