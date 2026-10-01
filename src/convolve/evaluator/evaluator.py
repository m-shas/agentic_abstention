from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from convolve.config import Settings
from convolve.cost.controller import CostController
from convolve.evaluator.features import extract_features, heuristic_stop_probability
from convolve.evaluator.model import EvaluatorArtifact, load_or_none
from convolve.types import Decision, Reason, Trajectory

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluatorVerdict:
    decision: Decision
    stop_probability: float
    base_threshold: float
    threshold_used: float
    reason: Reason


class AbstentionEvaluator:
    """Scores a trajectory and returns a continue / abstain verdict.

    The evaluator is stateless. It loads a trained artefact if present and
    otherwise falls back to the deterministic heuristic. In both cases the
    final score is compared against a threshold that the cost controller may
    have lowered.
    """

    def __init__(
        self,
        settings: Settings,
        cost_controller: CostController,
        artifact: EvaluatorArtifact | None = None,
    ) -> None:
        self._settings = settings
        self._cost = cost_controller
        self._artifact = artifact if artifact is not None else load_or_none(
            settings.evaluator_model_path
        )
        if self._artifact is None:
            log.info("evaluator: using heuristic scorer (no trained artefact found)")
        else:
            log.info(
                "evaluator: loaded trained artefact (trained_on=%s)",
                self._artifact.metadata.get("n_samples", "unknown"),
            )

    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        return "trained" if self._artifact is not None else "heuristic"

    @property
    def artifact(self) -> EvaluatorArtifact | None:
        return self._artifact

    def reload(self, path: Path | None = None) -> None:
        p = path or self._settings.evaluator_model_path
        self._artifact = load_or_none(p)
        log.info("evaluator reloaded: source=%s", self.source)

    # ------------------------------------------------------------------

    def score(self, traj: Trajectory) -> float:
        features = extract_features(traj)
        if self._artifact is not None:
            return self._artifact.predict_one(features)
        return heuristic_stop_probability(features)

    def evaluate(self, traj: Trajectory) -> EvaluatorVerdict:
        """Decide whether to continue or abstain *before* the next action runs.

        Precedence:
          1. max steps already reached  -> abstain
          2. cost controller forces 0.0 -> abstain
          3. stop probability >= threshold -> abstain
          4. otherwise continue
        """
        base_threshold = traj.threshold

        # (1) hard step limit
        if traj.n_steps >= traj.max_steps:
            return EvaluatorVerdict(
                decision=Decision.ABSTAIN,
                stop_probability=1.0,
                base_threshold=base_threshold,
                threshold_used=base_threshold,
                reason=Reason.MAX_STEPS_REACHED,
            )

        stop_probability = self.score(traj)

        # (2) cost adjustment
        threshold_used, cost_forced = self._cost.adjust(
            base_threshold=base_threshold,
            spent=traj.spent,
            budget=traj.budget,
        )

        if cost_forced:
            return EvaluatorVerdict(
                decision=Decision.ABSTAIN,
                stop_probability=stop_probability,
                base_threshold=base_threshold,
                threshold_used=threshold_used,
                reason=Reason.COST_BUDGET_REACHED,
            )

        # (3) uncertainty threshold
        if stop_probability >= threshold_used:
            return EvaluatorVerdict(
                decision=Decision.ABSTAIN,
                stop_probability=stop_probability,
                base_threshold=base_threshold,
                threshold_used=threshold_used,
                reason=Reason.UNCERTAINTY_THRESHOLD,
            )

        return EvaluatorVerdict(
            decision=Decision.CONTINUE,
            stop_probability=stop_probability,
            base_threshold=base_threshold,
            threshold_used=threshold_used,
            reason=Reason.UNDER_THRESHOLD,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def build_abstention_message(traj: Trajectory, verdict: EvaluatorVerdict) -> str:
        if verdict.reason is Reason.MAX_STEPS_REACHED:
            return (
                f"I stopped after {traj.n_steps} steps because the maximum step budget "
                f"was reached. I could not confirm that the task '{traj.task}' is "
                f"completable with the available tools."
            )
        if verdict.reason is Reason.COST_BUDGET_REACHED:
            return (
                f"I stopped because the configured cost budget of "
                f"{traj.budget:.2f} was exhausted after {traj.n_steps} steps "
                f"(spent {traj.spent:.4f}). The task '{traj.task}' may still be "
                f"possible with a larger budget."
            )
        return (
            f"I cannot complete the task '{traj.task}'. After {traj.n_steps} steps my "
            f"estimated probability that further actions will succeed is "
            f"{1.0 - verdict.stop_probability:.0%}, which falls below the required "
            f"confidence. The available tools did not produce progress toward the goal."
        )
