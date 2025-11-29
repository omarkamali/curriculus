"""
Iterable dataset with curriculum learning support.
"""

import bisect
import itertools
import random
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .planner import CurriculusPlanner


class _CurriculusSplit:
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
                Additional recognised kwargs:
                    seed: Optional[int] to make iteration deterministic.
                    _transforms: Internal use – transformations to apply lazily.
        """

        seed = planner_kwargs.pop("seed", None)
        transforms = planner_kwargs.pop("_transforms", None)

        if planner is None:
            self.planner = CurriculusPlanner(datasets, **planner_kwargs)
        else:
            self.planner = planner

        self.datasets = datasets
        self.dataset_names = [d["name"] for d in datasets]
        self._transforms: List[Callable[[Any, int], Any]] = (
            list(transforms) if transforms is not None else []
        )

        self._base_seed = seed if seed is not None else random.randrange(2**63)
        self._schedule_breakpoints = [pct for pct, _ in self.planner.schedule]
        self._preview_cache: Dict[int, List[Any]] = {}
        self._cached_columns: Optional[List[str]] = None
        self.current_step = 0

    def _clone(
        self,
        *,
        transforms: Optional[List[Callable[[Any, int], Any]]] = None,
        seed: Optional[int] = None,
    ) -> "_CurriculusSplit":
        cloned_transforms = list(self._transforms) if transforms is None else list(transforms)
        clone = _CurriculusSplit(
            self.datasets,
            planner=self.planner,
            _transforms=cloned_transforms,
            seed=self._base_seed if seed is None else seed,
        )
        return clone

    def _reset_caches(self) -> None:
        self._preview_cache.clear()
        self._cached_columns = None

    def _create_iterators(self) -> Dict[str, Iterator[Any]]:
        iterators: Dict[str, Iterator[Any]] = {}
        for name in self.dataset_names:
            ds = self.planner.dataset_map[name]
            try:
                base_iterator: Iterable[Any] = ds
            except TypeError as exc:  # pragma: no cover - defensive branch
                raise TypeError(f"Dataset '{name}' is not iterable: {ds!r}") from exc

            if self.planner.oversampling:
                iterators[name] = itertools.cycle(base_iterator)
            else:
                iterators[name] = iter(base_iterator)

        return iterators

    def _apply_transforms(self, item: Any, index: int) -> Any:
        transformed = item
        for transform in self._transforms:
            transformed = transform(transformed, index)
        return transformed

    def _get_current_weights(
        self,
        step: Optional[int] = None,
        exhausted: Optional[set[str]] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Calculate interpolated weights for a given step.

        Args:
            step: Current iteration step.
            exhausted: Optional set tracking exhausted dataset names.

        Returns:
            (dataset_names, probabilities) tuple for weighted sampling
        """

        exhausted = exhausted or set()

        effective_step = self.current_step if step is None else step
        total_steps = max(self.planner.total_steps, 1)
        progress = max(0.0, min(effective_step / total_steps, 1.0))

        sched = self.planner.schedule

        idx = bisect.bisect_right(self._schedule_breakpoints, progress) - 1
        idx = max(0, min(idx, len(sched) - 2))

        s_pct, s_w = sched[idx]
        e_pct, e_w = sched[idx + 1]

        alpha = (progress - s_pct) / (e_pct - s_pct) if e_pct > s_pct else 0.0

        raw_weights: Dict[str, float] = {}
        active_keys = set(s_w.keys()) | set(e_w.keys())

        for key in active_keys:
            base_prob = s_w.get(key, 0.0) + alpha * (e_w.get(key, 0.0) - s_w.get(key, 0.0))
            factor = self.planner.scale_factors.get(key, 1.0)
            final_prob = base_prob * factor
            if final_prob > 0.0 and key not in exhausted:
                raw_weights[key] = final_prob

        total = sum(raw_weights.values())
        if total <= 0:
            return [], []

        keys = list(raw_weights.keys())
        probs = [raw_weights[k] / total for k in keys]

        return keys, probs

    def __iter__(self) -> Iterator[Any]:
        """Iterate over samples from the curriculum."""

        iterators = self._create_iterators()
        exhausted: set[str] = set()
        rng = random.Random(self._base_seed)

        produced = 0
        self.current_step = 0

        while produced < self.planner.total_steps:
            keys, probs = self._get_current_weights(self.current_step, exhausted)
            if not keys:
                break

            chosen_key = rng.choices(keys, weights=probs, k=1)[0]
            iterator = iterators.get(chosen_key)

            if iterator is None:
                exhausted.add(chosen_key)
                continue

            try:
                item = next(iterator)
            except StopIteration:
                exhausted.add(chosen_key)
                iterators[chosen_key] = None
                continue

            yield self._apply_transforms(item, produced)
            produced += 1
            self.current_step += 1

    def __len__(self) -> int:
        """Return total number of steps."""
        return self.planner.total_steps

    # ------------------------------------------------------------------
    # Convenience utilities & metadata
    # ------------------------------------------------------------------

    def with_seed(self, seed: Optional[int]) -> "_CurriculusSplit":
        """Return a clone configured with a new base seed."""

        new_seed = random.randrange(2**63) if seed is None else seed
        return self._clone(seed=new_seed)

    def peek(self, n: int = 5) -> List[Any]:
        """Return the first ``n`` examples without exhausting the dataset."""

        if n <= 0:
            return []

        if n not in self._preview_cache:
            self._preview_cache[n] = list(itertools.islice(iter(self), n))

        return list(self._preview_cache[n])

    def head(self, n: int = 5) -> List[Any]:
        """Alias for :meth:`peek` for pandas-style familiarity."""

        return self.peek(n)

    def take(self, n: int) -> List[Any]:
        """Convenience method mirroring Hugging Face APIs."""

        return self.peek(n)

    def to_list(self, limit: Optional[int] = None) -> List[Any]:
        """Materialise the dataset into a list."""

        iterator = iter(self)
        if limit is None:
            return list(iterator)
        return list(itertools.islice(iterator, limit))

    def to_pandas(self, limit: Optional[int] = None):  # pragma: no cover - optional dep
        """Convert the dataset (optionally truncated) to a pandas.DataFrame."""

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise ImportError(
                "pandas is required for to_pandas(); install with `pip install pandas`."
            ) from exc

        data = self.to_list(limit)
        return pd.DataFrame(data)

    def to_hf_iterable_dataset(self):  # pragma: no cover - optional dep
        """Materialize the curriculum as a Hugging Face ``IterableDataset``."""

        try:
            from datasets import IterableDataset
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise ImportError(
                "Hugging Face `datasets` is required. Install with `pip install datasets`."
            ) from exc

        def generator() -> Iterator[Any]:
            for item in self:
                yield item

        return IterableDataset.from_generator(generator)

    def to_hf_dataset(self, limit: Optional[int] = None):  # pragma: no cover - optional dep
        """Materialize the curriculum as a Hugging Face :class:`datasets.Dataset`."""

        try:
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise ImportError(
                "Hugging Face `datasets` is required. Install with `pip install datasets`."
            ) from exc

        data = self.to_list(limit)
        if not data:
            return Dataset.from_list([])

        first = data[0]
        if not isinstance(first, Mapping):
            raise TypeError(
                "to_hf_dataset requires dict-like examples. Consider map() to convert records first."
            )

        return Dataset.from_list([dict(item) for item in data])

    # ------------------------------------------------------------------
    # Column metadata & transformations
    # ------------------------------------------------------------------

    def _ensure_mapping(self, item: Any, op: str) -> Dict[str, Any]:
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError(
            f"{op} requires dict-like examples, but received {type(item).__name__}."
        )

    def _invalidate_metadata(self) -> None:
        self._reset_caches()

    @property
    def columns(self) -> List[str]:
        """Return discovered column names from a sample example."""

        if self._cached_columns is None:
            preview = self.peek(1)
            if not preview:
                self._cached_columns = []
            else:
                first = preview[0]
                if isinstance(first, Mapping):
                    self._cached_columns = list(first.keys())
                else:
                    self._cached_columns = []

        return list(self._cached_columns)

    @property
    def column_names(self) -> List[str]:
        """Alias for :pyattr:`columns`."""

        return self.columns

    @property
    def num_columns(self) -> int:
        """Return the number of detected columns."""

        return len(self.columns)

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        """Lightweight approximation mirroring Hugging Face datasets."""

        return len(self), self.num_columns or None

    def remove_columns(
        self,
        column_names: Sequence[str] | str,
        *,
        copy: bool = True,
    ) -> "_CurriculusSplit":
        names = [column_names] if isinstance(column_names, str) else list(column_names)

        def transform(item: Any, _: int) -> Any:
            mapping = self._ensure_mapping(item, "remove_columns")
            missing = [name for name in names if name not in mapping]
            if missing:
                raise KeyError(f"Columns not found for removal: {missing}")
            for name in names:
                mapping.pop(name, None)
            return mapping

        target: _CurriculusSplit = self if not copy else self._clone()
        target._transforms.append(transform)
        target._invalidate_metadata()
        return target

    def remove_column(
        self,
        column_name: str,
        *,
        copy: bool = True,
    ) -> "_CurriculusSplit":
        return self.remove_columns([column_name], copy=copy)

    def rename_columns(
        self,
        column_mapping: Mapping[str, str],
        *,
        copy: bool = True,
    ) -> "_CurriculusSplit":
        mapping = dict(column_mapping)

        def transform(item: Any, _: int) -> Any:
            data = self._ensure_mapping(item, "rename_columns")
            for old_name, new_name in mapping.items():
                if old_name not in data:
                    raise KeyError(f"Column '{old_name}' not found to rename")
                if new_name in data and new_name not in mapping:
                    raise KeyError(
                        f"Cannot rename '{old_name}' to '{new_name}': target already exists"
                    )
            for old_name, new_name in mapping.items():
                data[new_name] = data.pop(old_name)
            return data

        target: _CurriculusSplit = self if not copy else self._clone()
        target._transforms.append(transform)
        target._invalidate_metadata()
        return target

    def rename_column(
        self,
        original_column_name: str,
        new_column_name: str,
        *,
        copy: bool = True,
    ) -> "_CurriculusSplit":
        return self.rename_columns({original_column_name: new_column_name}, copy=copy)

    def map(  # pylint: disable=too-many-arguments
        self,
        function: Callable[..., Any],
        *,
        with_indices: bool = False,
        batched: bool = False,
        remove_columns: Optional[Sequence[str]] = None,
        copy: bool = True,
        **fn_kwargs: Any,
    ) -> "_CurriculusSplit":
        if batched:
            raise NotImplementedError("batched=True is not yet supported for curriculum datasets")

        remove_names = list(remove_columns or [])

        def transform(item: Any, idx: int) -> Any:
            args = (item, idx) if with_indices else (item,)
            result = function(*args, **fn_kwargs)
            if remove_names:
                mapping = self._ensure_mapping(result, "map/remove_columns")
                for name in remove_names:
                    mapping.pop(name, None)
                return mapping
            return result

        target: _CurriculusSplit = self if not copy else self._clone()
        target._transforms.append(transform)
        target._invalidate_metadata()
        return target

    def __repr__(self) -> str:  # pragma: no cover - repr is for debugging
        preview = self.peek(3)
        columns = self.columns
        preview_str = ", ".join(self._format_preview(item) for item in preview)
        return (
            f"{self.__class__.__name__}(total_steps={len(self)}, "
            f"columns={columns}, preview=[{preview_str}])"
        )

    def _format_preview(self, item: Any) -> str:
        text = repr(item)
        return text if len(text) <= 120 else f"{text[:117]}..."


class CurriculusSplits(Mapping[str, _CurriculusSplit]):
    """Mapping of curriculum splits to iterable datasets."""

    def __init__(self, splits: Dict[str, _CurriculusSplit]):
        self._splits = splits

    def __getitem__(self, key: str) -> _CurriculusSplit:
        return self._splits[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._splits)

    def __len__(self) -> int:
        return len(self._splits)

    def __repr__(self) -> str:  # pragma: no cover - repr is for debugging
        splits = ", ".join(self._splits.keys())
        return f"{self.__class__.__name__}({{{splits}}})"

    def to_hf_dataset(self, limit: Optional[int] = None):  # pragma: no cover - optional dep
        """Materialize all splits as a Hugging Face :class:`datasets.DatasetDict`."""

        try:
            from datasets import DatasetDict
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise ImportError(
                "Hugging Face `datasets` is required. Install with `pip install datasets`."
            ) from exc

        dataset_dict = {
            name: split.to_hf_dataset(limit)
            for name, split in self._splits.items()
        }

        return DatasetDict(dataset_dict)


def _clone_planner(base: CurriculusPlanner, *, total_steps: int) -> CurriculusPlanner:
    """Create a planner clone with identical configuration but new total steps."""

    return CurriculusPlanner(
        base.datasets,
        schedule=base.schedule,
        total_steps=total_steps,
        oversampling=base.oversampling,
        best_effort=base.best_effort,
    )


def Curriculus(
    datasets: List[Dict[str, Any]],
    planner: Optional[CurriculusPlanner] = None,
    *,
    train_ratio: Optional[float] = None,
    split_names: Tuple[str, str] = ("train", "test"),
    **planner_kwargs: Any,
) -> CurriculusSplits:
    """Build curriculum iterable dataset splits.

    Args:
        datasets: Dataset configuration list.
        planner: Optional preconfigured planner.
        train_ratio: Fraction of total steps allocated to the train split.
            Defaults to 1.0 (train only).
        split_names: Names for the train and test splits.
        **planner_kwargs: Additional planner arguments (schedule, total_steps, etc.).

    Returns:
        CurriculusSplits with at least a ``train`` split.
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

    splits: Dict[str, _CurriculusSplit] = {}

    if train_steps > 0:
        if train_steps == base_planner.total_steps and planner is not None:
            train_planner = base_planner
        elif train_steps == base_planner.total_steps and planner is None:
            train_planner = base_planner
        else:
            train_planner = _clone_planner(base_planner, total_steps=train_steps)

        splits[split_names[0]] = _CurriculusSplit(
            datasets,
            planner=train_planner,
        )

    if test_steps > 0:
        test_planner = _clone_planner(base_planner, total_steps=test_steps)
        splits[split_names[1]] = _CurriculusSplit(
            datasets,
            planner=test_planner,
        )

    if not splits:
        raise ValueError("Split configuration produced no datasets")

    return CurriculusSplits(splits)
