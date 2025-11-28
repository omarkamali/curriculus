"""
Iterable dataset with curriculum learning support.
"""

import bisect
import itertools
import random
from typing import Any, Dict, List, Optional, Tuple

from .planner import CurriculusPlanner


class CurriculusIterableDataset:
    """
    An iterable dataset that progressively mixes multiple datasets based on a schedule.

    This class yields samples from different datasets with probabilities that change
    over time according to the curriculum schedule.
    """

    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        planner: Optional[CurriculusPlanner] = None,
        **planner_kwargs: Any,
    ):
        """
        Initialize the curriculum dataset.

        Args:
            datasets: List of {name, dataset} dicts
            planner: Optional pre-configured CurriculusPlanner
            **planner_kwargs: Passed to CurriculusPlanner if planner is None
                            (schedule, total_steps, oversampling, best_effort)
        """
        if planner is None:
            self.planner = CurriculusPlanner(datasets, **planner_kwargs)
        else:
            self.planner = planner

        self.datasets = datasets
        self.dataset_names = [d["name"] for d in datasets]

        # Setup iterators
        cycle = self.planner.oversampling
        self.iterators: Dict[str, Any] = {}
        self.exhausted: set[str] = set()

        for name in self.dataset_names:
            ds = self.planner.dataset_map[name]
            if cycle:
                self.iterators[name] = itertools.cycle(ds)
            else:
                self.iterators[name] = iter(ds)

        self.current_step = 0

    def _get_current_weights(self) -> Tuple[List[str], List[float]]:
        """
        Calculate interpolated weights for current step.

        Returns:
            (dataset_names, probabilities) tuple for weighted sampling
        """
        progress = self.current_step / max(self.planner.total_steps, 1)
        progress = min(max(progress, 0.0), 1.0)

        sched = self.planner.schedule

        # Find interval containing current progress
        idx = bisect.bisect_right([x[0] for x in sched], progress) - 1
        idx = max(0, min(idx, len(sched) - 2))

        s_pct, s_w = sched[idx]
        e_pct, e_w = sched[idx + 1]

        # Linear interpolation
        if e_pct > s_pct:
            alpha = (progress - s_pct) / (e_pct - s_pct)
        else:
            alpha = 0.0

        raw_weights: Dict[str, float] = {}
        active_keys = set(s_w.keys()) | set(e_w.keys())

        for k in active_keys:
            base_prob = s_w.get(k, 0.0) + alpha * (e_w.get(k, 0.0) - s_w.get(k, 0.0))

            # Apply best-effort scaling
            factor = self.planner.scale_factors.get(k, 1.0)
            final_prob = base_prob * factor

            if final_prob > 0:
                raw_weights[k] = final_prob

        for exhausted_key in list(raw_weights.keys()):
            if exhausted_key in self.exhausted:
                raw_weights.pop(exhausted_key)

        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        if total <= 0:
            return [], []

        keys = list(raw_weights.keys())
        probs = [raw_weights[k] / total for k in keys]

        return keys, probs

    def __iter__(self) -> Any:
        """Iterate over samples from the curriculum."""
        while self.current_step < self.planner.total_steps:
            keys, probs = self._get_current_weights()
            if not keys:
                break

            chosen_key = random.choices(keys, weights=probs, k=1)[0]
            iterator = self.iterators.get(chosen_key)

            if iterator is None:
                self.exhausted.add(chosen_key)
                continue

            try:
                item = next(iterator)
            except StopIteration:
                self.exhausted.add(chosen_key)
                self.iterators[chosen_key] = None
                continue

            yield item
            self.current_step += 1

    def __len__(self) -> int:
        """Return total number of steps."""
        return self.planner.total_steps
