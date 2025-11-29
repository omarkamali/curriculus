"""
Iterable dataset with curriculum learning support.
"""

import bisect
import itertools
import random
from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .planner import CurriculusPlanner


class _CurriculusIterableDataset:
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

    def __iter__(self) -> Iterator[Any]:
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


class CurriculusIterableDatasetDict(Mapping[str, _CurriculusIterableDataset]):
    """Mapping of curriculum splits to iterable datasets."""

    def __init__(self, splits: Dict[str, _CurriculusIterableDataset]):
        self._splits = splits

    def __getitem__(self, key: str) -> _CurriculusIterableDataset:
        return self._splits[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._splits)

    def __len__(self) -> int:
        return len(self._splits)

    def __repr__(self) -> str:  # pragma: no cover - repr is for debugging
        splits = ", ".join(self._splits.keys())
        return f"{self.__class__.__name__}({{{splits}}})"


def _clone_planner(base: CurriculusPlanner, *, total_steps: int) -> CurriculusPlanner:
    """Create a planner clone with identical configuration but new total steps."""

    return CurriculusPlanner(
        base.datasets,
        schedule=base.schedule,
        total_steps=total_steps,
        oversampling=base.oversampling,
        best_effort=base.best_effort,
    )


def CurriculusIterableDataset(
    datasets: List[Dict[str, Any]],
    planner: Optional[CurriculusPlanner] = None,
    *,
    train_ratio: Optional[float] = None,
    split_names: Tuple[str, str] = ("train", "test"),
    **planner_kwargs: Any,
) -> CurriculusIterableDatasetDict:
    """Build curriculum iterable dataset splits.

    Args:
        datasets: Dataset configuration list.
        planner: Optional preconfigured planner.
        train_ratio: Fraction of total steps allocated to the train split.
            Defaults to 1.0 (train only).
        split_names: Names for the train and test splits.
        **planner_kwargs: Additional planner arguments (schedule, total_steps, etc.).

    Returns:
        CurriculusIterableDatasetDict with at least a ``train`` split.
    """

    if len(split_names) != 2:
        raise ValueError("split_names must contain exactly two entries (train, test)")

    ratio = 1.0 if train_ratio is None else float(train_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("train_ratio must be between 0.0 and 1.0 inclusive")

    base_planner = planner or CurriculusPlanner(datasets, **planner_kwargs)
    total_steps = base_planner.total_steps

    train_steps = int(round(total_steps * ratio))
    train_steps = max(0, min(train_steps, total_steps))
    test_steps = total_steps - train_steps

    splits: Dict[str, _CurriculusIterableDataset] = {}

    if train_steps > 0:
        if train_steps == base_planner.total_steps and planner is not None:
            train_planner = base_planner
        elif train_steps == base_planner.total_steps and planner is None:
            train_planner = base_planner
        else:
            train_planner = _clone_planner(base_planner, total_steps=train_steps)

        splits[split_names[0]] = _CurriculusIterableDataset(
            datasets,
            planner=train_planner,
        )

    if test_steps > 0:
        test_planner = _clone_planner(base_planner, total_steps=test_steps)
        splits[split_names[1]] = _CurriculusIterableDataset(
            datasets,
            planner=test_planner,
        )

    if not splits:
        raise ValueError("Split configuration produced no datasets")

    return CurriculusIterableDatasetDict(splits)
