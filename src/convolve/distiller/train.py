from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from convolve.distiller.dataset import TrajectoryDataset, build_dataset, load_trajectories
from convolve.distiller.decision_transformer import (
    DTConfig,
    build_model,
    is_available,
    num_parameters,
)

log = logging.getLogger(__name__)


@dataclass
class TrainReport:
    epochs: int
    train_loss: float
    val_loss: float
    n_train: int
    n_val: int
    params: int
    checkpoint: str


def _require_torch():
    if not is_available():
        raise RuntimeError(
            "PyTorch is required for training. Install with: pip install 'convolve[torch]'"
        )
    import torch

    return torch


def train(
    trajectories_path: Path,
    checkpoint_path: Path,
    *,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    val_frac: float = 0.2,
    max_len: int = 128,
    seed: int = 42,
    config: DTConfig | None = None,
) -> TrainReport:
    torch = _require_torch()

    records = load_trajectories(trajectories_path)
    if not records:
        raise FileNotFoundError(f"no trajectories found at {trajectories_path}")

    dataset = build_dataset(records, max_len=max_len)
    train_ds, val_ds = dataset.train_val_split(val_frac=val_frac, seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = config or DTConfig(max_len=max_len)
    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="mean")

    def make_batches(ds: TrajectoryDataset):
        n = len(ds)
        order = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            yield (
                torch.from_numpy(ds.tokens[idx]).long().to(device),
                torch.from_numpy(ds.rtg[idx]).float().to(device),
                torch.from_numpy(ds.targets[idx]).long().to(device),
                torch.from_numpy(ds.mask[idx]).float().to(device),
            )

    def run_epoch(training: bool) -> float:
        model.train(training)
        ds = train_ds if training else val_ds
        if len(ds) == 0:
            return 0.0
        total, count = 0.0, 0
        for toks, rtg, tgt, msk in make_batches(ds):
            if training:
                opt.zero_grad(set_to_none=True)
            logits = model(toks, rtg, msk)
            loss = loss_fn(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += float(loss.item())
            count += 1
        return total / max(1, count)

    train_loss = val_loss = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(True)
        val_loss = run_epoch(False)
        log.info("epoch %d/%d train=%.4f val=%.4f", epoch, epochs, train_loss, val_loss)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "meta": {
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "records": len(records),
            },
        },
        checkpoint_path,
    )

    return TrainReport(
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        n_train=len(train_ds),
        n_val=len(val_ds),
        params=num_parameters(model),
        checkpoint=str(checkpoint_path),
    )


def load_model(checkpoint_path: Path, device: str = "cpu"):
    torch = _require_torch()
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = DTConfig(**blob["config"])
    model = build_model(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg, blob.get("meta", {})
