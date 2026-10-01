# Training Guide

Convolve has two offline training stages. Both are optional: the service runs
with a heuristic evaluator and no playbooks out of the box.

## 0. Generate or collect trajectories

For a quick start:

```bash
convolve simulate --n 500
```

This writes `data/trajectories/trajectories.jsonl` with three outcome classes:
`success`, `abstained`, `looped`.

For real data, point Convolve at a running deployment and collect the JSONL
that `POST /v1/sessions/{id}/end` appends on every session close.

## 1. Train the abstention evaluator

```bash
convolve train-evaluator
```

This:

1. Loads every trajectory record.
2. Labels each with `1` if the outcome is `looped` or `aborted`, else `0`.
3. Extracts the 12 features from each trajectory.
4. Fits a `StandardScaler + GradientBoostingClassifier` pipeline.
5. Fits an isotonic calibrator on the raw scores.
6. Saves `artifacts/evaluator.joblib` and reports ECE before/after.

At startup the proxy automatically picks up the artefact. You can hot-reload
it with `POST /internal/reload-evaluator` if you add that route, or by
restarting the service.

### Inspecting the classifier

```python
from convolve.evaluator.model import EvaluatorArtifact

art = EvaluatorArtifact.load("artifacts/evaluator.joblib")
print(art.pipeline.named_steps["clf"].feature_importances_)
```

## 2. Train the Decision Transformer distiller

Requires PyTorch:

```bash
pip install -e ".[torch]"
convolve train-distiller --epochs 8 --batch-size 32 --lr 3e-4
```

The trainer:

1. Encodes each trajectory with a hashing tokenizer.
2. Computes per-step return-to-go from the session outcome using
   `rewards.return_to_go`. Terminal reward: `success=+1.0`,
   `abstained=+0.5`, `aborted=-0.3`, `error=-0.5`, `looped=-1.0`. Each step
   also pays a `0.05` cost penalty.
3. Trains a 2-layer, 4-head, 64-dim Decision Transformer to predict the next
   action token conditioned on the running return-to-go.
4. Saves `artifacts/distiller.pt`.

The model is small enough to train on CPU in a few minutes for thousands of
trajectories.

## 3. Mine playbooks

```bash
convolve extract-playbooks --max-playbooks 100 --clear
```

For every `looped`, `error`, or `aborted` trajectory, the trained DT is asked
at which step its ABSTAIN probability first exceeds `0.5`. That step becomes
the `abstain_at_step` for a new playbook. Rules are keyed by a coarse task
signature so that similar tasks share rules.

The retriever then ranks playbooks by (signature match, Jaccard similarity of
task tokens, stored score). The top match is appended to the abstention
message when a live session abstains.

## 4. Evaluate

```bash
convolve eval --threshold 0.7
```

This reports accuracy, precision, recall, and F1 over the labelled trajectory
set. Run it once for the heuristic and again after training the artefact to
see the improvement.

### Threshold sweep

```bash
for t in 0.3 0.5 0.7 0.9; do
  echo "threshold=$t"
  convolve eval --threshold "$t"
done
```

## Retraining loop

In production, the recommended cadence is:

1. Collect trajectories continuously via `POST /end`.
2. Nightly: `convolve train-evaluator`.
3. Weekly: `convolve train-distiller && convolve extract-playbooks --clear`.
4. Restart or hot-reload the proxy to pick up the new artefact.
