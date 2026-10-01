from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from convolve.evaluator.calibration import IsotonicCalibrator
from convolve.evaluator.features import FEATURE_ORDER, heuristic_stop_probability, to_matrix


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=42,
                ),
            ),
        ]
    )


@dataclass
class EvaluatorArtifact:
    """Everything needed to reproduce a trained evaluator's decisions."""

    pipeline: Pipeline
    calibrator: IsotonicCalibrator
    feature_order: tuple[str, ...] = FEATURE_ORDER
    metadata: dict = field(default_factory=dict)

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.raw_scores(X))

    def predict_one(self, features: dict[str, float]) -> float:
        if any(k not in features for k in self.feature_order):
            missing = [k for k in self.feature_order if k not in features]
            raise KeyError(f"missing features: {missing}")
        X = to_matrix([features])
        return float(self.predict(X)[0])

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "calibrator": self.calibrator,
                "feature_order": list(self.feature_order),
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> EvaluatorArtifact:
        blob = joblib.load(path)
        return cls(
            pipeline=blob["pipeline"],
            calibrator=blob["calibrator"],
            feature_order=tuple(blob["feature_order"]),
            metadata=blob.get("metadata", {}),
        )

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        metadata: dict | None = None,
    ) -> EvaluatorArtifact:
        if len(np.unique(y)) < 2:
            raise ValueError("training labels must contain both classes")
        pipeline = build_pipeline()
        pipeline.fit(X, y)
        raw = pipeline.predict_proba(X)[:, 1]
        cal = IsotonicCalibrator().fit(raw, y)
        return cls(pipeline=pipeline, calibrator=cal, metadata=metadata or {})


def load_or_none(path: Path) -> EvaluatorArtifact | None:
    if path.exists():
        try:
            return EvaluatorArtifact.load(path)
        except Exception:
            return None
    return None


__all__ = [
    "EvaluatorArtifact",
    "build_pipeline",
    "load_or_none",
    "heuristic_stop_probability",
]
