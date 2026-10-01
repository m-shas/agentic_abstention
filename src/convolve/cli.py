from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from convolve.config import get_settings
from convolve.logging_setup import configure_logging

app = typer.Typer(
    name="convolve",
    help="Convolve — agentic abstention and sequential uncertainty middleware.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host (default: settings.host)"),
    port: int | None = typer.Option(None, help="Bind port (default: settings.port)"),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn auto-reload"),
) -> None:
    """Run the Convolve interceptor proxy."""
    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "convolve.proxy.app:app_factory",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


# ---------------------------------------------------------------------------
# init-db
# ---------------------------------------------------------------------------


@app.command("init-db")
def init_db() -> None:
    """Apply Alembic migrations."""
    from alembic.config import Config

    from alembic import command

    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(cfg, "head")
    console.print("[green]Database migrated to head.[/green]")


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


@app.command()
def simulate(
    n: int = typer.Option(200, help="Number of synthetic trajectories"),
    out: Path | None = typer.Option(None, help="Output directory"),
    seed: int = typer.Option(7, help="Random seed"),
) -> None:
    """Generate synthetic trajectories for offline training."""
    from scripts.simulate_trajectories import generate

    settings = get_settings()
    out_dir = out or settings.trajectory_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trajectories.jsonl"
    written = generate(n=n, seed=seed, path=path)
    console.print(f"[green]Wrote {written} trajectories to {path}[/green]")


# ---------------------------------------------------------------------------
# train-evaluator
# ---------------------------------------------------------------------------


@app.command("train-evaluator")
def train_evaluator(
    trajectories: Path | None = typer.Option(None, help="Path to trajectories JSONL/dir"),
    out: Path | None = typer.Option(None, help="Where to save the artefact"),
) -> None:
    """Fit and calibrate the abstention classifier from labelled trajectories."""
    import numpy as np

    from convolve.distiller.dataset import load_trajectories
    from convolve.evaluator.calibration import expected_calibration_error
    from convolve.evaluator.features import extract_features, to_matrix
    from convolve.evaluator.model import EvaluatorArtifact
    from convolve.types import Step, Trajectory

    settings = get_settings()
    traj_path = trajectories or settings.trajectory_dir
    out_path = out or settings.evaluator_model_path

    records = load_trajectories(traj_path)
    if not records:
        console.print(f"[red]No trajectories found at {traj_path}[/red]")
        raise typer.Exit(code=1)

    X_rows: list[dict] = []
    y_rows: list[int] = []

    for rec in records:
        steps = rec.get("steps", [])
        if not steps:
            continue
        outcome = rec.get("outcome", "error")
        label = 1 if outcome in ("looped", "aborted") else 0
        traj = Trajectory(
            session_id=rec.get("session_id", ""),
            task=rec.get("task", ""),
            max_steps=rec.get("max_steps", 12),
            budget=rec.get("budget", 0.5),
            threshold=rec.get("threshold", 0.7),
            spent=rec.get("spent", 0.0),
            steps=[
                Step(
                    idx=s.get("idx", i),
                    thought=s.get("thought", ""),
                    action=s.get("action", ""),
                    action_input=s.get("action_input", {}),
                    observation=s.get("observation"),
                    estimated_cost=s.get("estimated_cost", 0.0),
                    actual_cost=s.get("actual_cost", 0.0),
                    is_error=s.get("is_error", False),
                )
                for i, s in enumerate(steps)
            ],
        )
        X_rows.append(extract_features(traj))
        y_rows.append(label)

    X = to_matrix(X_rows)
    y = np.asarray(y_rows, dtype=int)

    if len(np.unique(y)) < 2:
        console.print(
            "[yellow]Only one class present in the dataset. "
            "Generate more diverse trajectories (include looped ones).[/yellow]"
        )
        raise typer.Exit(code=1)

    artifact = EvaluatorArtifact.fit(
        X, y, metadata={"n_samples": int(len(y)), "source": str(traj_path)}
    )

    raw = artifact.pipeline.predict_proba(X)[:, 1]
    calibrated = artifact.predict(X)
    ece_before = expected_calibration_error(y, raw)
    ece_after = expected_calibration_error(y, calibrated)

    artifact.save(out_path)
    console.print(f"[green]Saved evaluator artefact to {out_path}[/green]")

    table = Table(title="Evaluator training summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Samples", str(len(y)))
    table.add_row("Positive (impossible)", str(int(y.sum())))
    table.add_row("Negative (solvable)", str(int((y == 0).sum())))
    table.add_row("ECE before calibration", f"{ece_before:.4f}")
    table.add_row("ECE after calibration", f"{ece_after:.4f}")
    console.print(table)


# ---------------------------------------------------------------------------
# train-distiller
# ---------------------------------------------------------------------------


@app.command("train-distiller")
def train_distiller(
    trajectories: Path | None = typer.Option(None, help="Path to trajectories JSONL/dir"),
    out: Path | None = typer.Option(None, help="Checkpoint output path"),
    epochs: int = typer.Option(5, help="Training epochs"),
    batch_size: int = typer.Option(16, help="Batch size"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
) -> None:
    """Train the Decision Transformer on logged trajectories."""
    from convolve.distiller.decision_transformer import is_available
    from convolve.distiller.train import train

    if not is_available():
        console.print(
            "[red]PyTorch is not installed. Install with: pip install 'convolve[torch]'[/red]"
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    traj_path = trajectories or settings.trajectory_dir
    out_path = out or settings.distiller_checkpoint

    report = train(
        trajectories_path=traj_path,
        checkpoint_path=out_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    table = Table(title="Distiller training summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Epochs", str(report.epochs))
    table.add_row("Train samples", str(report.n_train))
    table.add_row("Val samples", str(report.n_val))
    table.add_row("Train loss", f"{report.train_loss:.4f}")
    table.add_row("Val loss", f"{report.val_loss:.4f}")
    table.add_row("Parameters", f"{report.params:,}")
    table.add_row("Checkpoint", report.checkpoint)
    console.print(table)


# ---------------------------------------------------------------------------
# extract-playbooks
# ---------------------------------------------------------------------------


@app.command("extract-playbooks")
def extract_playbooks_cmd(
    trajectories: Path | None = typer.Option(None),
    checkpoint: Path | None = typer.Option(None),
    max_playbooks: int = typer.Option(100),
    clear: bool = typer.Option(False, "--clear", help="Delete existing playbooks first"),
) -> None:
    """Mine stopping rules from failing trajectories and store them."""
    from convolve.db.session import get_database
    from convolve.distiller.decision_transformer import is_available
    from convolve.distiller.playbook import extract_playbooks, persist_playbooks
    from convolve.playbooks.store import PlaybookStore

    if not is_available():
        console.print("[red]PyTorch is required for this command.[/red]")
        raise typer.Exit(code=1)

    settings = get_settings()
    traj_path = trajectories or settings.trajectory_dir
    ckpt = checkpoint or settings.distiller_checkpoint

    if not ckpt.exists():
        console.print(f"[red]Checkpoint not found: {ckpt}[/red]")
        raise typer.Exit(code=1)

    async def _run() -> int:
        db = get_database()
        store = PlaybookStore(db)
        if clear:
            await store.clear()
        extracted = extract_playbooks(
            trajectories_path=traj_path,
            checkpoint_path=ckpt,
            store=store,
            max_playbooks=max_playbooks,
        )
        count = await persist_playbooks(store, extracted)
        await db.dispose()
        return count

    count = asyncio.run(_run())
    console.print(f"[green]Stored {count} playbooks.[/green]")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@app.command("eval")
def evaluate(
    trajectories: Path | None = typer.Option(None),
    model: Path | None = typer.Option(None),
    threshold: float | None = typer.Option(None, help="Override decision threshold"),
) -> None:
    """Evaluate the abstention evaluator on a labelled trajectory set."""

    from convolve.config import Settings
    from convolve.cost.controller import CostController
    from convolve.distiller.dataset import load_trajectories
    from convolve.evaluator.evaluator import AbstentionEvaluator
    from convolve.types import Step, Trajectory

    settings = get_settings()
    traj_path = trajectories or settings.trajectory_dir
    model_path = model or settings.evaluator_model_path

    if not model_path.exists():
        console.print(
            f"[yellow]No trained evaluator found at {model_path}; using heuristic.[/yellow]"
        )

    cfg = Settings(
        evaluator_model_path=model_path,
        default_threshold=threshold if threshold is not None else settings.default_threshold,
    )
    cost = CostController(knee=cfg.cost_knee)
    evaluator = AbstentionEvaluator(cfg, cost)

    records = load_trajectories(traj_path)
    if not records:
        console.print("[red]No trajectories to evaluate.[/red]")
        raise typer.Exit(code=1)

    tp = fp = tn = fn = 0
    for rec in records:
        traj = Trajectory(
            session_id=rec.get("session_id", ""),
            task=rec.get("task", ""),
            max_steps=rec.get("max_steps", 12),
            budget=rec.get("budget", 0.5),
            threshold=threshold if threshold is not None else rec.get("threshold", 0.7),
            spent=rec.get("spent", 0.0),
            steps=[
                Step(
                    idx=s.get("idx", i),
                    thought=s.get("thought", ""),
                    action=s.get("action", ""),
                    action_input=s.get("action_input", {}),
                    observation=s.get("observation"),
                    is_error=s.get("is_error", False),
                )
                for i, s in enumerate(rec.get("steps", []))
            ],
        )
        verdict = evaluator.evaluate(traj)
        truly_impossible = rec.get("outcome") in ("looped", "aborted")
        stopped = verdict.decision.value == "abstain"
        if truly_impossible and stopped:
            tp += 1
        elif truly_impossible and not stopped:
            fn += 1
        elif (not truly_impossible) and stopped:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    table = Table(title="Abstention evaluation")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Evaluator source", evaluator.source)
    table.add_row("Threshold", f"{cfg.default_threshold:.2f}")
    table.add_row("Total trajectories", str(total))
    table.add_row("True positives (correct stops)", str(tp))
    table.add_row("False positives (wrong stops)", str(fp))
    table.add_row("True negatives (correct continues)", str(tn))
    table.add_row("False negatives (missed stops)", str(fn))
    table.add_row("Accuracy", f"{accuracy:.4f}")
    table.add_row("Precision", f"{precision:.4f}")
    table.add_row("Recall", f"{recall:.4f}")
    table.add_row("F1", f"{f1:.4f}")
    console.print(table)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the Convolve version."""
    from convolve.version import __version__

    console.print(json.dumps({"convolve": __version__}))


if __name__ == "__main__":  # pragma: no cover
    app()
