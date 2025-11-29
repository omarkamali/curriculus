"""Tests for curriculum dataset."""

import random

import pytest
from curriculus import Curriculus


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, name, data=None):
        self.name = name
        self.data = data or [name] * 100

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class TestCurriculus:
    """Test curriculum iterable splits."""

    def test_basic_iteration(self):
        """Test basic iteration over curriculum dataset."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 100)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 100)},
        ]
        ds_dict = Curriculus(configs, total_steps=10, oversampling=True)

        items = list(ds_dict["train"])
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
        ds_dict = Curriculus(
            configs, schedule=sched, total_steps=1000, oversampling=True
        )

        items = list(ds_dict["train"])
        assert len(items) == 1000

        segments = [items[i : i + 250] for i in range(0, 1000, 250)]
        observed = [sum(1 for item in seg if item == "B") / len(seg) for seg in segments]
        expected = [0.0, 0.0, 0.25, 0.75]

        for idx, (obs, exp) in enumerate(zip(observed, expected)):
            assert obs == pytest.approx(exp, abs=0.1), f"Quarter {idx + 1}"

    def test_len(self):
        """Test __len__ returns total_steps."""
        configs = [{"name": "A", "dataset": MockDataset("A")}]  # pragma: no mutate
        ds_dict = Curriculus(configs, total_steps=100)
        assert len(ds_dict["train"]) == 100

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
        ds_dict = Curriculus(
            configs, schedule=sched, total_steps=1000, oversampling=True
        )

        # At step 500 (progress=0.5), weights should be approximately 50/50
        train_split = ds_dict["train"]
        train_split.current_step = 500
        keys, probs = train_split._get_current_weights()

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
        ds_dict = Curriculus(configs, oversampling=True)

        # Should auto-generate schedule and total_steps
        train_split = ds_dict["train"]
        assert len(train_split.planner.schedule) == 2
        assert train_split.planner.total_steps > 0

    def test_get_current_weights_at_boundaries(self):
        """Test weight calculation at schedule boundaries."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 100)},
        ]
        sched = [(0.0, {"A": 1.0}), (1.0, {"A": 1.0})]
        ds_dict = Curriculus(
            configs, schedule=sched, total_steps=100, oversampling=True
        )

        # At step 0
        train_split = ds_dict["train"]
        train_split.current_step = 0
        keys, probs = train_split._get_current_weights()
        assert keys == ["A"]
        assert probs == [1.0]

        # At step 99 (last step)
        train_split.current_step = 99
        keys, probs = train_split._get_current_weights()
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
        ds_dict = Curriculus(
            configs,
            schedule=sched,
            total_steps=200,
            oversampling=False,
            best_effort=True,
        )

        # With best effort, B's probability should be reduced
        train_split = ds_dict["train"]
        keys, probs = train_split._get_current_weights()
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
        ds_dict = Curriculus(
            configs,
            schedule=sched,
            total_steps=200,
            oversampling=False,
            best_effort=True,
        )

        items = list(ds_dict["train"])
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
        ds_dict = Curriculus(
            configs,
            schedule=sched,
            total_steps=200,
            oversampling=True,
            best_effort=False,
        )

        items = list(ds_dict["train"])
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
            Curriculus(
                configs,
                schedule=sched,
                total_steps=200,
                oversampling=False,
                best_effort=False,
            )

    def test_train_test_split_ratio(self):
        """Dataset dict exposes train/test splits respecting ratio."""
        configs = [
            {"name": "A", "dataset": MockDataset("A", ["A"] * 100)},
            {"name": "B", "dataset": MockDataset("B", ["B"] * 100)},
        ]

        ds_dict = Curriculus(
            configs, total_steps=100, oversampling=True, train_ratio=0.8
        )

        assert set(ds_dict.keys()) == {"train", "test"}
        assert len(ds_dict["train"]) == 80
        assert len(ds_dict["test"]) == 20

    def test_columns_metadata_and_repr(self):
        """Iterable dataset exposes column metadata and readable repr."""
        configs = [
            {
                "name": "A",
                "dataset": MockDataset(
                    "A",
                    [
                        {"x": 1, "y": 2},
                        {"x": 3, "y": 4},
                        {"x": 5, "y": 6},
                    ],
                ),
            }
        ]

        ds_dict = Curriculus(configs, total_steps=3)
        train_split = ds_dict["train"]

        assert train_split.columns == ["x", "y"]
        assert train_split.column_names == ["x", "y"]
        assert train_split.num_columns == 2
        assert train_split.shape == (3, 2)

        preview = train_split.peek(2)
        assert preview == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

        representation = repr(train_split)
        assert "columns=['x', 'y']" in representation
        assert "total_steps=3" in representation

    def test_remove_and_rename_columns(self):
        """Column removal and renaming operate lazily on examples."""
        configs = [
            {
                "name": "A",
                "dataset": MockDataset(
                    "A",
                    [
                        {"x": 1, "y": 2},
                        {"x": 3, "y": 4},
                    ],
                ),
            }
        ]

        train_split = Curriculus(configs, total_steps=2)["train"]

        trimmed = train_split.remove_column("y")
        trimmed_rows = trimmed.to_list()
        assert trimmed_rows == [{"x": 1}, {"x": 3}]
        assert trimmed.columns == ["x"]

        renamed = train_split.rename_columns({"x": "feature_x", "y": "feature_y"})
        renamed_rows = renamed.to_list()
        assert renamed_rows == [
            {"feature_x": 1, "feature_y": 2},
            {"feature_x": 3, "feature_y": 4},
        ]
        assert renamed.columns == ["feature_x", "feature_y"]

        # Original split should remain unchanged because operations default to copy=True
        original_rows = train_split.to_list()
        assert original_rows == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

    def test_map_with_indices_and_remove_columns(self):
        """Map supports index-aware transforms and column pruning."""
        configs = [
            {
                "name": "A",
                "dataset": MockDataset(
                    "A",
                    [
                        {"x": 10, "y": 1},
                        {"x": 20, "y": 2},
                        {"x": 30, "y": 3},
                    ],
                ),
            }
        ]

        train_split = Curriculus(configs, total_steps=3)["train"]

        mapped = train_split.map(lambda ex, idx: {**ex, "idx": idx}, with_indices=True)
        mapped_rows = mapped.take(3)
        assert mapped_rows == [
            {"x": 10, "y": 1, "idx": 0},
            {"x": 20, "y": 2, "idx": 1},
            {"x": 30, "y": 3, "idx": 2},
        ]

        summed = train_split.map(
            lambda ex: {**ex, "sum": ex["x"] + ex["y"]}, remove_columns=["y"]
        )
        summed_rows = summed.to_list()
        assert summed_rows == [
            {"x": 10, "sum": 11},
            {"x": 20, "sum": 22},
            {"x": 30, "sum": 33},
        ]

    def test_to_hf_dataset_requires_mapping(self):
        """to_hf_dataset converts mapping examples and rejects non-mapping samples."""
        pytest.importorskip("datasets")
        configs = [
            {
                "name": "A",
                "dataset": MockDataset(
                    "A",
                    [
                        {"x": 1, "y": 2},
                        {"x": 3, "y": 4},
                    ],
                ),
            }
        ]

        train_split = Curriculus(configs, total_steps=2)["train"]
        ds = train_split.to_hf_dataset()
        assert ds.num_rows == 2
        assert set(ds.column_names) == {"x", "y"}

        list_split = Curriculus(
            [
                {
                    "name": "B",
                    "dataset": MockDataset("B", [1, 2, 3]),
                }
            ],
            total_steps=3,
        )["train"]

        with pytest.raises(TypeError, match="dict-like"):
            list_split.to_hf_dataset()

    def test_splits_to_hf_datasetdict(self):
        """CurriculusSplits materializes into HF DatasetDict."""
        datasets_mod = pytest.importorskip("datasets")

        configs = [
            {
                "name": "A",
                "dataset": MockDataset(
                    "A",
                    [
                        {"value": 1},
                        {"value": 2},
                    ],
                ),
            }
        ]

        splits = Curriculus(configs, total_steps=2)
        ds_dict = splits.to_hf_dataset()
        assert isinstance(ds_dict, datasets_mod.DatasetDict)
        assert ds_dict["train"].num_rows == 2

    def test_mixed_schema_preserves_union(self):
        """Mixing datasets with different schemas retains union of columns."""

        configs = [
            {
                "name": "rich",
                "dataset": MockDataset(
                    "rich",
                    [
                        {"x": 1, "y": 1},
                        {"x": 2, "y": 4},
                    ],
                ),
            },
            {
                "name": "lean",
                "dataset": MockDataset(
                    "lean",
                    [
                        {"x": 10},
                        {"x": 20},
                    ],
                ),
            },
        ]

        schedule = [
            (0.0, {"rich": 1.0, "lean": 0.0}),
            (0.5, {"rich": 1.0, "lean": 0.0}),
            (0.75, {"rich": 0.0, "lean": 1.0}),
            (1.0, {"rich": 0.0, "lean": 1.0}),
        ]

        splits = Curriculus(
            configs,
            schedule=schedule,
            total_steps=4,
            oversampling=True,
        )

        # Force deterministic sampling order for the regression.
        train_split = splits["train"].with_seed(0)

        preview = train_split.peek(4)
        assert len(preview) == 4
        assert train_split.columns == ["x", "y"]

        lean_rows = [row for row in preview if row["x"] in {10, 20}]
        assert lean_rows
        assert all(row["y"] is None for row in lean_rows)

    def test_shuffled_false_preserves_original_order(self):
        """Default iteration preserves original dataset order when not shuffled."""

        data = list(range(10))
        configs = [
            {
                "name": "ordered",
                "dataset": data,
            }
        ]

        splits = Curriculus(
            configs,
            total_steps=len(data),
            seed=123,
            shuffled=False,
        )

        items = list(splits["train"])
        assert items == data

    def test_shuffled_true_uses_deterministic_order(self):
        """Shuffled datasets honour shuffle_seed for deterministic permutations."""

        data = list(range(10))
        configs = [
            {
                "name": "to_shuffle",
                "dataset": data,
            }
        ]

        shuffle_seed = 9876
        splits = Curriculus(
            configs,
            total_steps=len(data),
            shuffled=True,
            shuffle_seed=shuffle_seed,
        )

        items = list(splits["train"])

        expected = list(data)
        rng = random.Random(shuffle_seed)
        rng.shuffle(expected)
        assert items == expected

    def test_union_of_columns_across_three_datasets(self):
        """Datasets with progressive schemas expose the full union of columns."""

        configs = [
            {
                "name": "one",
                "dataset": [
                    {"a": 1},
                    {"a": 2},
                ],
            },
            {
                "name": "two",
                "dataset": [
                    {"a": 10, "b": 20},
                    {"a": 11, "b": 21},
                ],
            },
            {
                "name": "three",
                "dataset": [
                    {"a": 100, "b": 200, "c": 300},
                    {"a": 101, "b": 201, "c": 301},
                ],
            },
        ]

        schedule = [
            (0.0, {"one": 1.0}),
            (0.4, {"one": 1.0}),
            (0.6, {"two": 1.0}),
            (0.8, {"two": 1.0}),
            (1.0, {"three": 1.0}),
        ]

        splits = Curriculus(
            configs,
            schedule=schedule,
            total_steps=12,
            oversampling=True,
            seed=0,
        )

        train_split = splits["train"]
        rows = train_split.to_list()

        for row in rows:
            assert set(row.keys()) == {"a", "b", "c"}

        assert any(row["b"] is not None for row in rows)
        assert any(row["c"] is not None for row in rows)
        assert train_split.columns == ["a", "b", "c"]

    def test_curriculus_accepts_positional_schedule(self):
        """Providing schedule positionally should configure the planner."""

        configs = [
            {
                "name": "A",
                "dataset": MockDataset("A", ["A"] * 10),
            },
            {
                "name": "B",
                "dataset": MockDataset("B", ["B"] * 10),
            },
        ]

        schedule = [
            (0.0, {"A": 1.0, "B": 0.0}),
            (1.0, {"A": 0.0, "B": 1.0}),
        ]

        splits = Curriculus(configs, schedule, total_steps=4, oversampling=True)
        items = list(splits["train"])
        assert len(items) == 4
        assert set(items).issubset({"A", "B"})

    def test_curriculus_rejects_planner_argument(self):
        """Providing a planner argument is no longer supported."""

        configs = [
            {
                "name": "A",
                "dataset": MockDataset("A", ["A"] * 5),
            }
        ]

        schedule = [
            (0.0, {"A": 1.0}),
            (1.0, {"A": 1.0}),
        ]

        with pytest.raises(TypeError):
            Curriculus(configs, schedule, planner=None)  # type: ignore[arg-type]

    def test_columns_union_primed_from_features(self):
        """Column metadata is primed from dataset features without iteration."""

        class HFStub:
            def __init__(self, rows, features):
                self._rows = rows
                self.features = features

            def __iter__(self):
                yield from self._rows

            def __len__(self):
                return len(self._rows)

        configs = [
            {
                "name": "foundational",
                "dataset": HFStub(
                    [{"messages": 1}],
                    ["messages"],
                ),
            },
            {
                "name": "translation",
                "dataset": HFStub(
                    [{"messages": 2, "token_count": 10}],
                    ["messages", "token_count"],
                ),
            },
            {
                "name": "instruct",
                "dataset": HFStub(
                    [{"messages": 3, "token_count": 20, "tools": 1}],
                    ["messages", "token_count", "tools"],
                ),
            },
        ]

        split = Curriculus(configs, total_steps=3, oversampling=True)["train"]

        assert split.columns == ["messages", "token_count", "tools"]

        sample = split.peek(1)[0]
        assert set(sample.keys()) == {"messages", "token_count", "tools"}

    def test_union_of_columns_auto_schedule(self):
        """Auto-generated schedules still retain the full column union."""

        configs = [
            {
                "name": "one",
                "dataset": [
                    {"a": 1},
                ],
            },
            {
                "name": "two",
                "dataset": [
                    {"a": 10, "b": 20},
                ],
            },
            {
                "name": "three",
                "dataset": [
                    {"a": 100, "b": 200, "c": 300},
                ],
            },
        ]

        splits = Curriculus(
            configs,
            total_steps=9,
            oversampling=True,
            seed=0,
        )

        train_split = splits["train"]
        rows = train_split.to_list()

        for row in rows:
            assert set(row.keys()) == {"a", "b", "c"}

        assert any(row["b"] is not None for row in rows)
        assert any(row["c"] is not None for row in rows)
        assert train_split.columns == ["a", "b", "c"]

