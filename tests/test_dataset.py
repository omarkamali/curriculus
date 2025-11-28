"""Tests for curriculum dataset."""

import random

import pytest
from curriculus.dataset import CurriculusIterableDataset


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, name, data=None):
        self.name = name
        self.data = data or [name] * 100

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class TestCurriculusIterableDataset:
    """Test the iterable dataset."""

    def test_basic_iteration(self):
        """Test basic iteration over curriculum dataset."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 100)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 100)},
        ]
        ds = CurriculusIterableDataset(configs, total_steps=10, oversampling=True)

        items = list(ds)
        assert len(items) == 10
        assert all(item in ["A", "B"] for item in items)

    def test_sampling_respects_schedule(self):
        """Test that sampling respects the schedule."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 500)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 500)},
        ]
        sched = [
            (0.0, {"A": 1.0, "B": 0.0}),
            (0.5, {"A": 1.0, "B": 0.0}),
            (1.0, {"A": 0.0, "B": 1.0}),
        ]
        random.seed(1234)
        ds = CurriculusIterableDataset(
            configs, schedule=sched, total_steps=1000, oversampling=True
        )

        items = list(ds)
        assert len(items) == 1000

        segments = [items[i : i + 250] for i in range(0, 1000, 250)]
        observed = [sum(1 for item in seg if item == "B") / len(seg) for seg in segments]
        expected = [0.0, 0.0, 0.25, 0.75]

        for idx, (obs, exp) in enumerate(zip(observed, expected)):
            assert obs == pytest.approx(exp, abs=0.1), f"Quarter {idx + 1}"

    def test_len(self):
        """Test __len__ returns total_steps."""
        configs = [{"name": "A", "dataset": MockDataset("A")}]  # pragma: no mutate
        ds = CurriculusIterableDataset(configs, total_steps=100)
        assert len(ds) == 100

    def test_interpolation_at_mid_schedule(self):
        """Test weight interpolation at midpoint."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 1000)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 1000)},
        ]
        # A: 1.0 -> 0.0 over 0->1
        # B: 0.0 -> 1.0 over 0->1
        sched = [
            (0.0, {"A": 1.0, "B": 0.0}),
            (1.0, {"A": 0.0, "B": 1.0}),
        ]
        ds = CurriculusIterableDataset(
            configs, schedule=sched, total_steps=1000, oversampling=True
        )

        # At step 500 (progress=0.5), weights should be approximately 50/50
        ds.current_step = 500
        keys, probs = ds._get_current_weights()

        assert set(keys) == {"A", "B"}
        # Probabilities should be close to [0.5, 0.5]
        assert 0.4 < probs[0] < 0.6
        assert 0.4 < probs[1] < 0.6

    def test_auto_configuration(self):
        """Test auto-configuration via kwargs."""
        configs = [
            {"name": "A", "dataset": MockDataset("A")},
            {"name": "B", "dataset": MockDataset("B")},
        ]
        ds = CurriculusIterableDataset(configs, oversampling=True)

        # Should auto-generate schedule and total_steps
        assert len(ds.planner.schedule) == 2
        assert ds.planner.total_steps > 0

    def test_get_current_weights_at_boundaries(self):
        """Test weight calculation at schedule boundaries."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 100)},
        ]
        sched = [(0.0, {"A": 1.0}), (1.0, {"A": 1.0})]
        ds = CurriculusIterableDataset(
            configs, schedule=sched, total_steps=100, oversampling=True
        )

        # At step 0
        ds.current_step = 0
        keys, probs = ds._get_current_weights()
        assert keys == ["A"]
        assert probs == [1.0]

        # At step 99 (last step)
        ds.current_step = 99
        keys, probs = ds._get_current_weights()
        assert keys == ["A"]
        assert probs == [1.0]

    def test_best_effort_scaling(self):
        """Test that best effort scaling affects weights."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 100)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 50)},
        ]
        # Both asked equally, but B only has 50% of data
        sched = [
            (0.0, {"A": 0.5, "B": 0.5}),
            (1.0, {"A": 0.5, "B": 0.5}),
        ]
        # B is short, should be scaled down (factor ~0.5)
        ds = CurriculusIterableDataset(
            configs,
            schedule=sched,
            total_steps=200,
            oversampling=False,
            best_effort=True,
        )

        # With best effort, B's probability should be reduced
        keys, probs = ds._get_current_weights()
        prob_map = dict(zip(keys, probs))
        assert set(prob_map) == {"A", "B"}
        # A should have more than B since B was scaled down (expected 2/3 vs 1/3)
        assert prob_map["A"] == pytest.approx(2 / 3, abs=0.05)
        assert prob_map["B"] == pytest.approx(1 / 3, abs=0.05)

    def test_best_effort_scaling_limits_sampling(self):
        """Iteration respects best-effort scaling when data is short."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 200)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 50)},
        ]
        sched = [
            (0.0, {"A": 0.5, "B": 0.5}),
            (1.0, {"A": 0.5, "B": 0.5}),
        ]
        random.seed(4321)
        ds = CurriculusIterableDataset(
            configs,
            schedule=sched,
            total_steps=200,
            oversampling=False,
            best_effort=True,
        )

        items = list(ds)
        assert len(items) == 200
        ratio_b = items.count("B") / len(items)
        # Only 50 B samples exist, so best-effort scaling should consume all
        # of them, yielding ~25% of total output.
        assert ratio_b == pytest.approx(0.25, abs=0.05)
        assert ratio_b < 0.5  # ensure scaling reduced B usage

    def test_oversampling_repeats_data_when_short(self):
        """Oversampling ensures mix is preserved even when data is short."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 200)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 50)},
        ]
        sched = [
            (0.0, {"A": 0.5, "B": 0.5}),
            (1.0, {"A": 0.5, "B": 0.5}),
        ]
        random.seed(2468)
        ds = CurriculusIterableDataset(
            configs,
            schedule=sched,
            total_steps=200,
            oversampling=True,
            best_effort=False,
        )

        items = list(ds)
        assert len(items) == 200
        ratio_b = items.count("B") / len(items)
        assert ratio_b == pytest.approx(0.5, abs=0.08)

    def test_strict_mode_raises_on_shortage(self):
        """Strict mode without oversampling should raise when data is insufficient."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 200)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 50)},
        ]
        sched = [
            (0.0, {"A": 0.5, "B": 0.5}),
            (1.0, {"A": 0.5, "B": 0.5}),
        ]

        with pytest.raises(ValueError, match="shortage"):
            CurriculusIterableDataset(
                configs,
                schedule=sched,
                total_steps=200,
                oversampling=False,
                best_effort=False,
            )
