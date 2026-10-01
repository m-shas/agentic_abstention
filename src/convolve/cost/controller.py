from __future__ import annotations


class CostController:
    """Lowers the abstention threshold as a fraction of the budget is consumed.

    Behaviour
    ---------
    * spending <= knee * budget          -> threshold unchanged
    * knee * budget < spending < budget  -> threshold falls linearly to 0
    * spending >= budget                 -> threshold = 0 and `forced` is True
    """

    def __init__(self, knee: float = 0.5) -> None:
        if not 0.0 <= knee < 1.0:
            raise ValueError("knee must be in [0, 1)")
        self._knee = knee

    @property
    def knee(self) -> float:
        return self._knee

    def adjusted_threshold(self, base_threshold: float, spent: float, budget: float) -> float:
        threshold, _ = self.adjust(base_threshold, spent, budget)
        return threshold

    def adjust(
        self, base_threshold: float, spent: float, budget: float
    ) -> tuple[float, bool]:
        """Return (threshold_to_use, forced_by_budget)."""
        if budget <= 0.0:
            return base_threshold, False

        fraction = max(0.0, spent) / budget

        if fraction >= 1.0:
            return 0.0, True

        if fraction <= self._knee:
            return base_threshold, False

        # Linear decay from base_threshold at `knee` to 0.0 at 1.0.
        span = 1.0 - self._knee
        progress = (fraction - self._knee) / span  # in (0, 1)
        return max(0.0, base_threshold * (1.0 - progress)), False

    def remaining(self, spent: float, budget: float) -> float:
        if budget <= 0.0:
            return float("inf")
        return max(0.0, budget - spent)
