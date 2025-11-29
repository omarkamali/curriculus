"""
Curriculus: Progressive curriculum learning for LLM training.

This package provides tools to gradually mix and transition between different
datasets during training using linear interpolation between probability schedules.
"""

__version__ = "0.1.4"

from .planner import CurriculusPlanner, generate_sequential_schedule
from .dataset import Curriculus, CurriculusSplits

__all__ = [
    "CurriculusPlanner",
    "Curriculus",
    "CurriculusSplits",
    "generate_sequential_schedule",
]
