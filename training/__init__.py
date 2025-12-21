# Training module for Smart Grocery Tracker
# Contains loss functions and training utilities

from .criterion import SetCriterion, build_criterion
from .trainer import Trainer

__all__ = [
    "SetCriterion",
    "build_criterion",
    "Trainer",
]
