"""Treehopper: Structure-preserving subsampling via greedy k-centers."""

__version__ = "0.2.0"

from .hoppers import hopper, treehopper, PCATreePartition, RPartition
from .utils import haus, haus_curve

__all__ = [
    "hopper",
    "treehopper",
    "PCATreePartition",
    "RPartition",
    "haus",
    "haus_curve",
]
