"""Utility functions for evaluating sketch quality."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked

logger = logging.getLogger(__name__)


def haus(
    data: np.ndarray,
    sample: Sequence[int],
    max_mem: float = float("inf"),
) -> float:
    """Compute the one-sided Hausdorff distance from ``data`` to ``data[sample]``.

    Parameters
    ----------
    data : np.ndarray
        Full dataset (n_samples, n_features).
    sample : sequence of int
        Indices of the sketch points.
    max_mem : float
        If finite, use chunked computation with this working memory (MB).

    Returns
    -------
    float
        The maximum over all points of the distance to the nearest sketch point.
    """
    if max_mem == float("inf"):
        dist = pairwise_distances(data[sample, :], data, n_jobs=-1)
        return dist.min(0).max()
    else:
        dists = pairwise_distances_chunked(
            data,
            data[sample, :],
            reduce_func=lambda x, y: x.min(1),
            working_memory=max_mem,
        )
        h = 0
        for x in dists:
            h = max(h, max(x))
        return h


def haus_curve(
    data: np.ndarray,
    ordering: Sequence[int],
    distfunc=euclidean,
    max_len: int = 5000,
) -> list[float]:
    """Compute the Hausdorff distance curve as points are added in order.

    Parameters
    ----------
    data : np.ndarray
        Full dataset.
    ordering : sequence of int
        Order in which points are added to the sketch.
    distfunc : callable
        Distance metric.
    max_len : int
        Maximum number of points to evaluate.

    Returns
    -------
    list of float
        Hausdorff distance after adding each successive point.
    """
    result = []
    min_dists = np.full(data.shape[0], float("inf"))

    for i, idx in enumerate(ordering):
        if i > max_len:
            break
        new = data[idx, :]
        new_dists = pairwise_distances(new.reshape(1, -1), data).flatten()
        min_dists = np.minimum(min_dists, new_dists)
        result.append(float(min_dists.max()))

    return result
