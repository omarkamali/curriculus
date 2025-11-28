"""Tests for curriculum planner."""

import pytest
from curriculus.planner import CurriculusPlanner, generate_sequential_schedule


class TestGenerateSequentialSchedule:
    """Test the schedule generation utility."""

    def test_single_dataset(self):
        """Single dataset should have constant weight."""
        sched = generate_sequential_schedule(["A"])
        assert len(sched) == 2
        assert sched[0] == (0.0, {"A": 1.0})
        assert sched[1] == (1.0, {"A": 1.0})

    def test_two_datasets(self):
        """Two datasets should fade one into the other."""
        sched = generate_sequential_schedule(["A", "B"])
        assert len(sched) == 2
        assert sched[0] == (0.0, {"A": 1.0, "B": 0.0})
        assert sched[1] == (1.0, {"A": 0.0, "B": 1.0})

    def test_three_datasets(self):
        """Three datasets should create peaks at 0, 0.5, 1.0."""
        sched = generate_sequential_schedule(["A", "B", "C"])
        assert len(sched) == 3
        assert sched[0] == (0.0, {"A": 1.0, "B": 0.0, "C": 0.0})
        assert sched[1] == (0.5, {"A": 0.0, "B": 1.0, "C": 0.0})
        assert sched[2] == (1.0, {"A": 0.0, "B": 0.0, "C": 1.0})


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, name, size, data=None):
        self.name = name
        self._size = size
        self.data = data or [f"{name}_{i}" for i in range(size)]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self.data)


class TestCurriculusPlanner:
    """Test the curriculum planner."""

    def test_sequential_auto_schedule(self):
        """Test auto-generation of sequential schedule."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 1000)},
            {"name": "B", "dataset": MockDataset("B", 1000)},
        ]
        planner = CurriculusPlanner(configs, oversampling=False, best_effort=True)

        assert planner.total_steps == 2000
        assert len(planner.schedule) == 2
        assert planner.scale_factors["A"] == 1.0
        assert planner.scale_factors["B"] == 1.0

    def test_auto_steps_calculation(self):
        """Auto-calculate steps from dataset sizes."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 500)},
            {"name": "B", "dataset": MockDataset("B", 300)},
        ]
        planner = CurriculusPlanner(configs)
        assert planner.total_steps == 800

    def test_best_effort_scaling(self):
        """Test best effort scaling when data is short."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 1000)},
            {"name": "B", "dataset": MockDataset("B", 500)},
        ]
        # Sequential: A integral = 0.5, B integral = 0.5
        # Total steps = 1500
        # Needed A = 750 (available 1000) -> OK
        # Needed B = 750 (available 500) -> SHORT!
        # Factor B = 500/750 = 0.667
        planner = CurriculusPlanner(configs, oversampling=False, best_effort=True)

        assert planner.total_steps == 1500
        assert planner.scale_factors["A"] == 1.0
        assert 0.66 < planner.scale_factors["B"] < 0.67

    def test_oversampling_enabled(self):
        """Test that oversampling allows repeating data."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 1000)},
            {"name": "B", "dataset": MockDataset("B", 500)},
        ]
        planner = CurriculusPlanner(configs, oversampling=True, best_effort=False)

        assert planner.total_steps == 1500
        assert planner.scale_factors["A"] == 1.0
        assert planner.scale_factors["B"] == 1.0  # No scaling, just repeat

    def test_strict_mode_error(self):
        """Test that strict mode raises error on shortage."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 1000)},
            {"name": "B", "dataset": MockDataset("B", 500)},
        ]
        with pytest.raises(ValueError, match="shortage"):
            CurriculusPlanner(configs, oversampling=False, best_effort=False)

    def test_manual_total_steps(self):
        """Test explicit total_steps override."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 1000)},
            {"name": "B", "dataset": MockDataset("B", 1000)},
        ]
        planner = CurriculusPlanner(
            configs, total_steps=500, oversampling=False, best_effort=True
        )

        assert planner.total_steps == 500
        # Both datasets have plenty, so no scaling needed
        assert planner.scale_factors["A"] == 1.0
        assert planner.scale_factors["B"] == 1.0

    def test_custom_schedule(self):
        """Test using a custom schedule."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 1000)},
            {"name": "B", "dataset": MockDataset("B", 1000)},
        ]
        custom_sched = [
            (0.0, {"A": 1.0, "B": 0.0}),
            (1.0, {"A": 0.0, "B": 1.0}),
        ]
        planner = CurriculusPlanner(
            configs, schedule=custom_sched, total_steps=2000, oversampling=False
        )

        assert planner.schedule == custom_sched
        assert planner.total_steps == 2000

    def test_schedule_validation(self):
        """Test that invalid schedules raise errors."""
        configs = [{"name": "A", "dataset": MockDataset("A", 1000)}]

        # Schedule doesn't start at 0.0
        with pytest.raises(ValueError, match="must start at 0.0"):
            CurriculusPlanner(configs, schedule=[(0.5, {"A": 1.0}), (1.0, {"A": 1.0})])

        # Schedule doesn't end at 1.0
        with pytest.raises(ValueError, match="must end at 1.0"):
            CurriculusPlanner(configs, schedule=[(0.0, {"A": 1.0}), (0.5, {"A": 1.0})])

        # Weights don't sum to 1.0
        with pytest.raises(ValueError, match="must sum to"):
            CurriculusPlanner(
                configs, schedule=[(0.0, {"A": 0.5}), (1.0, {"A": 0.5})]
            )

    def test_integrals_calculation(self):
        """Test that integrals are calculated correctly."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 1000)},
            {"name": "B", "dataset": MockDataset("B", 1000)},
        ]
        sched = [
            (0.0, {"A": 1.0, "B": 0.0}),
            (0.5, {"A": 1.0, "B": 0.0}),
            (1.0, {"A": 0.0, "B": 1.0}),
        ]
        planner = CurriculusPlanner(configs, schedule=sched)

        # A: 0.5 * 0.5 (rect) + 0.5 * 0.5 (trapezoid avg) = 0.375 + 0.125 = 0.5
        # Actually: A goes 1->1 over 0->0.5 (area 0.5), then 1->0 over 0.5->1 (area 0.5)
        # Wait: 0->0.5: avg(1,1)=1, dur=0.5 -> 0.5
        # 0.5->1: avg(1,0)=0.5, dur=0.5 -> 0.25
        # A integral = 0.75
        assert 0.74 < planner.dataset_integrals["A"] < 0.76
        assert 0.24 < planner.dataset_integrals["B"] < 0.26

    def test_get_plan_summary(self):
        """Test that plan summary is generated."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", 1000)},
            {"name": "B", "dataset": MockDataset("B", 500)},
        ]
        planner = CurriculusPlanner(configs, oversampling=False, best_effort=True)

        summary = planner.get_plan_summary()
        assert "Total Steps:" in summary
        assert "Dataset Budget:" in summary
        assert "A:" in summary
        assert "B:" in summary
