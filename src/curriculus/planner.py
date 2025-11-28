"""
Curriculum planner for calculating schedules and validating dataset availability.
"""

from typing import Dict, List, Optional, Tuple, Any


def generate_sequential_schedule(dataset_names: List[str]) -> List[Tuple[float, Dict[str, float]]]:
    """
    Generate a sequential crossfade schedule where datasets fade one into the next.

    Example:
        >>> schedule = generate_sequential_schedule(["A", "B", "C"])
        >>> schedule[0]
        (0.0, {'A': 1.0, 'B': 0.0, 'C': 0.0})
        >>> schedule[-1]
        (1.0, {'A': 0.0, 'B': 0.0, 'C': 1.0})

    Args:
        dataset_names: List of dataset names in order of appearance

    Returns:
        List of (progress_percent, {dataset: weight}) tuples
    """
    n = len(dataset_names)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, {dataset_names[0]: 1.0}), (1.0, {dataset_names[0]: 1.0})]

    schedule = []
    for i, name in enumerate(dataset_names):
        pct = i / (n - 1)
        weights = {k: (1.0 if k == name else 0.0) for k in dataset_names}
        schedule.append((pct, weights))

    return schedule


class CurriculusPlanner:
    """
    Validates curriculum schedules and calculates training budget for each dataset.

    Ensures that datasets have sufficient samples to fulfill the schedule and
    provides scaling factors for best-effort sampling if needed.
    """

    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        schedule: Optional[List[Tuple[float, Dict[str, float]]]] = None,
        total_steps: Optional[int] = None,
        oversampling: bool = False,
        best_effort: bool = True,
    ):
        """
        Initialize the curriculum planner.

        Args:
            datasets: List of dicts with 'name' and 'dataset' keys.
                            Dataset can be object with __len__ or a size hint.
            schedule: List of (progress_percent, weights_dict) tuples.
                     If None, auto-generates sequential schedule.
            total_steps: Total training steps. If None, sums all dataset sizes.
            oversampling: If True, repeat datasets to fulfill schedule.
            best_effort: If True, scale down dataset usage if insufficient.
                        If False and insufficient data, raises error.

        Raises:
            ValueError: If validation fails and both oversampling and best_effort are False.
        """
        self.datasets = datasets
        self.dataset_map: Dict[str, Any] = {}
        self.dataset_sizes: Dict[str, int] = {}
        self.oversampling = oversampling
        self.best_effort = best_effort

        # Initialize and infer dataset sizes
        self._initialize_datasets()

        # Generate or use provided schedule
        dataset_names = [d["name"] for d in self.datasets]
        if schedule is None:
            self.schedule = generate_sequential_schedule(dataset_names)
        else:
            self.schedule = sorted(schedule, key=lambda x: x[0])

        # Validate and normalize schedule
        self._validate_schedule()

        # Infer total_steps if not provided
        if total_steps is None:
            known_size = sum(self.dataset_sizes.values())
            if known_size > 0:
                self.total_steps = known_size
            else:
                print("⚠️ No sizes or total_steps found. Defaulting to 1,000 steps.")
                self.total_steps = 1000
        else:
            self.total_steps = total_steps

        # Calculate integrals and budget factors
        self.dataset_integrals = self._calculate_integrals()
        self.scale_factors = self._calculate_budget_factors()

    def _initialize_datasets(self) -> None:
        """Load datasets and infer their sizes."""
        for config in self.datasets:
            name = config["name"]
            ds_ref = config["dataset"]

            # Handle string (repo ID) vs object
            if isinstance(ds_ref, str):
                # In real use, would do: load_dataset(ds_ref)
                # For now, we store the reference
                self.dataset_map[name] = ds_ref
            else:
                self.dataset_map[name] = ds_ref

            # Try to infer size
            if "size" in config:
                self.dataset_sizes[name] = config["size"]
            else:
                try:
                    self.dataset_sizes[name] = len(ds_ref)  # type: ignore
                except (TypeError, AttributeError):
                    pass  # Size unknown

    def _validate_schedule(self) -> None:
        """Validate that schedule is well-formed."""
        if not self.schedule:
            raise ValueError("Schedule cannot be empty")
        if self.schedule[0][0] != 0.0:
            raise ValueError("Schedule must start at 0.0")
        if self.schedule[-1][0] != 1.0:
            raise ValueError("Schedule must end at 1.0")

        # Check weights sum to 1.0
        for pct, weights in self.schedule:
            total_w = sum(weights.values())
            if not (0.99 <= total_w <= 1.01):
                raise ValueError(f"Weights at {pct} must sum to ~1.0 (got {total_w})")

    def _calculate_integrals(self) -> Dict[str, float]:
        """Calculate the area under each dataset's probability curve."""
        integrals: Dict[str, float] = {k: 0.0 for k in self.dataset_map.keys()}

        for i in range(len(self.schedule) - 1):
            s_pct, s_w = self.schedule[i]
            e_pct, e_w = self.schedule[i + 1]
            dur = e_pct - s_pct

            active = set(s_w.keys()) | set(e_w.keys())
            for k in active:
                avg = (s_w.get(k, 0.0) + e_w.get(k, 0.0)) / 2.0
                integrals[k] += avg * dur

        return integrals

    def _calculate_budget_factors(self) -> Dict[str, float]:
        """
        Calculate scaling factors for each dataset.

        Returns a factor ≤ 1.0 if best_effort is enabled and data is short.
        """
        factors: Dict[str, float] = {}

        for name, integral in self.dataset_integrals.items():
            if name not in self.dataset_sizes:
                # Unknown size, assume infinite
                factors[name] = 1.0
                continue

            needed = integral * self.total_steps
            available = self.dataset_sizes[name]

            if needed <= available or needed <= 0:
                factors[name] = 1.0
            else:
                # Shortage case
                if self.oversampling:
                    factors[name] = 1.0
                elif self.best_effort:
                    ratio = available / needed if needed > 0 else 0.0
                    factors[name] = ratio
                    print(
                        f"⚠️ WARNING: '{name}' shortage ({available} vs {int(needed)}). "
                        f"Scaling probability by {ratio:.2f}x (best_effort=True)."
                    )
                else:
                    raise ValueError(
                        f"❌ Dataset '{name}' shortage! Needed {int(needed)}, have {available}. "
                        f"Enable oversampling or best_effort."
                    )

        return factors

    def get_plan_summary(self) -> str:
        """Return a summary of the planned training."""
        lines = [
            f"Total Steps: {self.total_steps}",
            f"Oversampling: {self.oversampling}",
            f"Best Effort: {self.best_effort}",
            "Dataset Budget:",
        ]

        for name, integral in self.dataset_integrals.items():
            needed = int(integral * self.total_steps)
            available = self.dataset_sizes.get(name, 0)
            factor = self.scale_factors.get(name, 1.0)

            status = "OK"
            detail = f"{available} available"

            if needed > available:
                if factor < 1.0:
                    status = "SCALED"
                    detail = f"{available} available, {int(needed)} needed ({factor:.2f}x)"
                else:
                    status = "OVERSAMPLED"
                    detail = f"{available} available, {int(needed)} needed"

            lines.append(f"  {name}: {status:12} ({detail})")

        return "\\n".join(lines)
