"""
Curriculum: Progressive curriculum learning for LLM training.

This package provides tools to gradually mix and transition between different
datasets during training using linear interpolation between probability schedules.
"""

__version__ = "0.1.2"

from .planner import CurriculusPlanner, generate_sequential_schedule
from .dataset import CurriculusIterableDataset

__all__ = [
    "CurriculusPlanner",
    "CurriculusIterableDataset",
    "generate_sequential_schedule",
]
