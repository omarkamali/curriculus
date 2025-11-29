# Curriculus

[![PyPI version](https://img.shields.io/pypi/v/curriculus.svg)](https://pypi.org/project/curriculus/)
[![Test Matrix](https://github.com/omarkamali/curriculus/actions/workflows/pytest.yml/badge.svg)](https://github.com/omarkamali/curriculus/actions/workflows/pytest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Progressive curriculum learning for LLM training with fine-grained schedule control.

## What is this?

**Curriculus** helps you gradually mix and transition between different datasets during training. Instead of throwing all your data at a model at once, you can start with simpler data (e.g., basic easy), smoothly transition to more complex data (e.g., medium), and finally move to task-specific data (e.g., hard tuning).

The key insight: **linear interpolation between probability schedules**. You define milestones (e.g., "at 20%, start mixing medium in"), and the library handles the smooth transition with mathematically correct sampling.

## Why?

Training on progressively more complex data can:
- ✅ Improve model convergence and final performance
- ✅ Reduce training instability and catastrophic forgetting
- ✅ Allow precise control over when each dataset is used
- ✅ Handle datasets of different sizes gracefully

## Installation

```bash
pip install curriculus
```

With PyTorch support:
```bash
pip install curriculus[torch]
```

## Quick Start

### Minimal Example (Sequential Fading)

```python
from curriculus import Curriculus

# Your datasets
datasets = [
    {"name": "easy", "dataset": easy_data},
    {"name": "medium", "dataset": medium_data},
    {"name": "hard", "dataset": hard_data},
]

# Auto-generates: easy -> medium -> hard with train/test split
dataset_dict = Curriculus(datasets, train_ratio=0.8)

# Use with your trainer
for sample in dataset_dict["train"]:
    # sample comes from the appropriate dataset based on training progress
    pass
```

### Custom Schedule

```python
from curriculus import Curriculus

# Explicit schedule: define milestones and weights
schedule = [
    (0.0, {"easy": 1.0, "medium": 0.0, "hard": 0.0}),
    (0.2, {"easy": 1.0, "medium": 0.0, "hard": 0.0}),  # Warmup
    (0.4, {"easy": 0.5, "medium": 0.5, "hard": 0.0}),  # Easing
    (0.6, {"easy": 0.0, "medium": 1.0, "hard": 0.0}),  # Pure medium
    (0.8, {"easy": 0.0, "medium": 0.5, "hard": 0.5}),  # Mix
    (1.0, {"easy": 0.0, "medium": 0.0, "hard": 1.0}),  # Pure hard
]

dataset_dict = Curriculus(
    datasets,
    schedule=schedule,
    total_steps=10000,
    oversampling=True,  # Repeat data if insufficient
    best_effort=True,   # Scale down gracefully if short (default)
    train_ratio=0.9,    # 90% train, 10% test
)

# Access splits
train_data = dataset_dict["train"]
test_data = dataset_dict["test"]
```

## How It Works

### Schedule Interpretation

A schedule is a list of `(progress_percent, {dataset: weight})` tuples:

- **progress_percent** (0.0 to 1.0): Where you are in training
- **weight**: Probability of sampling from that dataset at this milestone

The library **linearly interpolates** between milestones. If you define:
- 0%: easy=1.0
- 100%: medium=1.0

Then at 50% progress, both have weight 0.5 (50/50 mix).

### Automatic Scale-Down (Best Effort)

If you don't have enough data:
- **best_effort=True** (default): Reduces the dataset's sampling probability to make it last
- **oversampling=True**: Repeats data to fulfill the schedule
- Both False: Raises an error

Example: If medium appears in the schedule but you only have 50% of the required samples:
- Best effort scales it down by 50%
- Other datasets naturally expand to fill the gap
- Training completes without crashing

### Dataset Sizes

Sizes are inferred automatically:
```python
datasets = [
    {"name": "A", "dataset": my_dataset},  # len() called automatically
]
```

Or specified manually:
```python
datasets = [
    {"name": "A", "dataset": huggingface_repo_id, "size": 50000},  # For streaming
]
```

## Configuration Options

### Curriculus

- **datasets**: List of `{"name": ..., "dataset": ...}` dicts.
- **schedule**: List of `(progress, weights)` tuples. If None, auto-generates sequential schedule.
- **total_steps**: Total training steps. If None, sums all dataset sizes.
- **oversampling**: If True, repeats data when insufficient. Default: False.
- **best_effort**: If True, scales down dataset usage gracefully. Default: True.
- **train_ratio**: Fraction of total steps for train split (0.0-1.0). Default: 1.0 (train only).
- **split_names**: Tuple of (train_name, test_name). Default: (`"train"`, `"test"`).

Returns:
    `CurriculusSplits` mapping of split names to iterable datasets.

Each split exposes convenient helpers to explore and transform the stream:
- `peek`/`head`/`take` preview upcoming samples without exhausting the iterator.
- `columns`, `shape`, and `num_columns` surface lightweight schema metadata.
- `remove_column` / `rename_column(s)` and `map` enable lazy columnar transforms.
- `to_hf_iterable_dataset()` and `to_hf_dataset()` materialise into Hugging Face
  `datasets.IterableDataset` or `datasets.Dataset` objects when you need the
  full HF toolkit.

## Real-World Example

```python
from curriculus import Curriculus

# Step 1: Load your datasets
easy_data = load_dataset("my_dataset/easy")
medium_data = load_dataset("my_dataset/medium")
hard_data = load_dataset("my_dataset/hard")

# Step 2: Define the curriculum
datasets = [
    {"name": "easy", "dataset": easy_data},
    {"name": "medium", "dataset": medium_data},
    {"name": "hard", "dataset": hard_data},
]

# Step 3: Create dataset with 85% train split
curriculum_dict = Curriculus(
    datasets,
    total_steps=100_000,
    oversampling=True,
    train_ratio=0.85
)

# Step 4: Use splits
for batch in DataLoader(curriculum_dict["train"], batch_size=32):
    loss = model.train_step(batch)

for batch in DataLoader(curriculum_dict["test"], batch_size=32):
    metrics = model.eval_step(batch)
```

## Advanced: Pre-flight Validation

Check your schedule without training:

```python
from curriculus import CurriculusPlanner

planner = CurriculusPlanner(
    datasets,
    schedule=my_schedule,
    total_steps=100_000,
    oversampling=False,
    best_effort=True,
)

print(planner.get_plan_summary())
# Output:
# Total Steps: 100000
# Dataset Budget:
#   easy: OK (1000000 available)
#   medium: SCALED (50000 available, 60000 needed (0.83x))
#   hard: OK (30000 available)
```

## Architecture

The library separates concerns:

- **CurriculusPlanner**: Validates schedules, calculates sample budgets, pre-flight checks
- **Curriculus**: Implements the actual sampling at training time

This allows you to validate your configuration before training starts, catching issues early.

## API Reference

### CurriculusPlanner

Validates and calculates sample budgets.

```python
planner = CurriculusPlanner(
    datasets,
    schedule=my_schedule,
    total_steps=100_000,
    oversampling=True,
    best_effort=True,
)

# Inspect
print(planner.scale_factors)  # Dict of scaling factors
print(planner.dataset_integrals)  # Area under each curve
print(planner.get_plan_summary())  # Human-readable plan
```

### Curriculus splits API

Iterates over mixed samples and exposes helpful adapters.

```python
dataset_splits = Curriculus(
    datasets,
    schedule=...,
    total_steps=100_000,
)

for sample in dataset_splits["train"]:
    # Sample is from the appropriate dataset based on progress
    pass

# Optional helpers
hf_iterable = dataset_splits["train"].to_hf_iterable_dataset()
hf_dataset = dataset_splits["train"].to_hf_dataset()

# Or directly on the dataset splits
hf_iterable = dataset_splits.to_hf_iterable_dataset()
hf_dataset = dataset_splits.to_hf_dataset()
```

### generate_sequential_schedule

Auto-generates a simple crossfade schedule. This function is called by default if you don't provide a schedule, and you will rarely need to use it directly.

```python
from curriculus import generate_sequential_schedule

schedule = generate_sequential_schedule(["dataset_A", "dataset_B", "dataset_C"])
# Result: A (100%) -> B (100%) -> C (100%)
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=curriculus

# View HTML coverage report
pytest --cov=curriculus --cov-report=html
# Open htmlcov/index.html
```

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Add tests for your changes
4. Ensure tests pass: `pytest --cov=curriculus`
5. Run linter: `ruff check --fix .`
6. Submit a pull request

## License

MIT License. See LICENSE file for details.

## Citation

If you use this library in research, please cite:

```bibtex
@software{curriculus2025,
  title={Curriculus: Progressive Curriculum Learning Datasets for LLM Training},
  author={Omar Kamali},
  year={2025},
  url={https://github.com/omarkamali/curriculus}
}
```

## Troubleshooting

### "Dataset 'X' shortage!"

You have more schedule demand than available data:
- **Solution 1**: Enable `best_effort=True` (default)
- **Solution 2**: Enable `oversampling=True`
- **Solution 3**: Increase dataset size or reduce `total_steps`

### Weights don't sum to 1.0

Your schedule is invalid:
```python
# ❌ Bad
schedule = [(0.0, {"A": 0.8, "B": 0.1})]  # Sum = 0.9

# ✅ Good
schedule = [(0.0, {"A": 0.8, "B": 0.2})]  # Sum = 1.0
```

### All samples from one dataset

Check that your schedule includes all datasets. If a dataset doesn't appear in the schedule, it's never sampled.

## Questions?

Open an issue: https://github.com/omarkamali/curriculus/issues


## Example Notebooks

Explore end-to-end walkthroughs in the `examples/` directory:

1. **Sequential difficulty fade** – [examples/01_easy_medium_hard.ipynb](examples/01_easy_medium_hard.ipynb)
2. **Conversation length autoschedule** – [examples/02_ultrachat_bucket_autoschedule.ipynb](examples/02_ultrachat_bucket_autoschedule.ipynb)
