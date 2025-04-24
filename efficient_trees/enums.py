"""
This module defines the enums for efficient trees.
"""

from enum import Enum


class Criterion(Enum):
    """
    Enum for splitting criterion
    """

    GINI = "gini"
    ENTROPY = "entropy"
