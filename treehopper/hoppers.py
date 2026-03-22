"""Core hopper and treehopper classes for structure-preserving subsampling."""

from __future__ import annotations

import itertools
import logging
import pickle
from collections import Counter
from copy import deepcopy
from functools import total_ordering
from time import time
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from fbpca import pca
from heapq import heappush, heappop
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


def RPartition(
    data: np.ndarray,
    max_partition_size: int = 1000,
    inds: Optional[List[int]] = None,
) -> List[List[int]]:
    """Median-partition along cycling dimensions until each partition is small.

    Splits data by comparing each dimension's median, cycling through dimensions,
    until every partition has at most ``max_partition_size`` points.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (n_samples, n_features).
    max_partition_size : int
        Maximum number of points per partition.
    inds : list of int, optional
        Row indices to partition. Defaults to all rows.

    Returns
    -------
    list of list of int
        Each inner list contains the row indices for one partition.
    """
    partitions = []
    if inds is None:
        inds = list(range(data.shape[0]))

    heappush(partitions, (-data.shape[0], 0, list(range(data.shape[0]))))
    current_partition = heappop(partitions)

    while len(current_partition[2]) > max_partition_size:
        dim = current_partition[1]
        rows = current_partition[2]
        vals = data[rows, dim].tolist()

        mid = np.median(vals)
        split = vals > mid

        p1 = list(itertools.compress(rows, split))
        p2 = list(itertools.compress(rows, 1 - split))

        newdim = (dim + 1) % data.shape[1]

        heappush(partitions, (-len(p1), newdim, p1))
        heappush(partitions, (-len(p2), newdim, p2))

        current_partition = heappop(partitions)

    heappush(partitions, current_partition)
    return [x[2] for x in partitions]


def PCATreePartition(
    data: np.ndarray,
    max_partition_size: int = 1000,
    inds: Optional[List[int]] = None,
) -> List[List[int]]:
    """Partition data by recursively splitting along the first principal component.

    At each step, the largest partition is projected onto its first PC and split
    at the median, producing two roughly equal halves. This continues until every
    partition has at most ``max_partition_size`` points.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (n_samples, n_features).
    max_partition_size : int
        Maximum number of points per partition.
    inds : list of int, optional
        Row indices to partition. Defaults to all rows.

    Returns
    -------
    list of list of int
        Each inner list contains the row indices for one partition.
    """
    partitions = []
    if inds is None:
        inds = list(range(data.shape[0]))

    heappush(partitions, (-data.shape[0], list(range(data.shape[0]))))
    current_partition = heappop(partitions)

    while len(current_partition[1]) > max_partition_size:
        rows = current_partition[1]
        subset = data[rows, :]

        U, s, Vt = pca(subset, k=1)
        pcvals = U[:, :1] * s[:1]

        mid = np.median(pcvals)
        split = pcvals > mid

        p1 = list(itertools.compress(rows, split))
        p2 = list(itertools.compress(rows, 1 - split))

        heappush(partitions, (-len(p1), p1))
        heappush(partitions, (-len(p2), p2))

        current_partition = heappop(partitions)

    heappush(partitions, current_partition)
    return [x[1] for x in partitions]


@total_ordering
class hopper:
    """Greedy k-centers subsampling via farthest-first traversal.

    Iteratively selects the point farthest from the current sketch, producing a
    2-approximation to the optimal Hausdorff distance between the sketch and the
    full dataset.

    Parameters
    ----------
    data : np.ndarray or None
        Input data matrix (n_samples, n_features).
    metric : callable
        Distance function compatible with ``sklearn.metrics.pairwise_distances``.
    inds : sequence of int, optional
        External index labels for each row of ``data``.
    start_r : float
        Initial covering radius (default ``inf``).
    root : int, optional
        Index of the starting point. If None, chosen at random.

    Attributes
    ----------
    path : list of int
        Row indices (into ``data``) of the sketch, in traversal order.
    path_inds : list of int
        External index labels of the sketch points.
    rs : list of float
        Covering radius after each hop.
    times : list of float
        Cumulative wall-clock time after each hop.
    vcells : list of int or None
        Voronoi cell assignment for every point in ``data``.
    """

    def __init__(
        self,
        data: Optional[np.ndarray],
        metric=euclidean,
        inds: Optional[Sequence[int]] = None,
        start_r: float = float("inf"),
        root: Optional[int] = None,
    ):
        t0 = time()
        self.times: List[float] = []
        self.r = start_r
        self.rs: List[float] = []

        if data is None:
            self.numObs = None
            self.numFeatures = None
        else:
            self.numObs, self.numFeatures = data.shape

        if inds is None:
            inds = range(self.numObs)

        self.inds = inds
        self.data = data
        self.path: List[int] = []
        self.path_inds: List[int] = []
        self.min_dists = []

        self.distfunc = metric
        self.vcells = None
        self.vdict = None
        self.wts = None

        self.root = root
        self.init_time = time() - t0
        self.times.append(self.init_time)
        self.new = True  # for treehopper

    def hop(self, n_hops: int = 1, store_vcells: bool = True) -> List[int]:
        """Add points to the sketch via farthest-first traversal.

        Parameters
        ----------
        n_hops : int
            Number of points to add.
        store_vcells : bool
            Whether to maintain Voronoi cell assignments.

        Returns
        -------
        list of int
            The full sketch path so far.
        """
        if self.data is None:
            raise ValueError("No data stored in this hopper!")

        for _ in itertools.repeat(None, n_hops):
            t0 = time()
            logger.info(f"Beginning traversal: {self.numObs} items to traverse")

            if len(self.path) == 0:
                if self.root is None:
                    first = np.random.choice(list(range(self.numObs)))
                else:
                    first = self.root

                self.path.append(first)
                self.path_inds.append(self.inds[first])

                first_pt = self.data[first, :].reshape((1, self.numFeatures))
                start_dists = pairwise_distances(
                    first_pt, self.data, metric=self.distfunc
                )[0, :]
                start_dists = np.array(start_dists)

                for ind in range(self.numObs):
                    if ind != first:
                        heappush(self.min_dists, (-start_dists[ind], ind))

                self.vcells = [self.inds[first]] * self.numObs
            else:
                if len(self.min_dists) < 1:
                    logger.info("Hopper exhausted!")
                    break

                next_ind = heappop(self.min_dists)[1]
                next_pt = self.data[next_ind, :].reshape((1, self.numFeatures))

                self.path.append(next_ind)
                self.path_inds.append(self.inds[next_ind])

                if store_vcells:
                    self.vcells[next_ind] = self.inds[next_ind]

                check_inds = []
                check_list = []
                prev_dists = []
                r = float("inf")

                if len(self.min_dists) > 0:
                    while r > self.r / 2 and len(self.min_dists) > 0:
                        curtuple = heappop(self.min_dists)
                        check_inds.append(curtuple[1])
                        check_list.append(curtuple)
                        prev_dists.append(-curtuple[0])
                        r = -curtuple[0]

                    heappush(self.min_dists, curtuple)
                    logger.debug(f"Checking {len(check_list)} points")

                    new_dists = pairwise_distances(
                        np.array(next_pt), self.data[check_inds, :],
                        metric=self.distfunc,
                    )[0, :]
                    new_dists = np.array(new_dists)

                    ischanged = new_dists < prev_dists
                    changed = list(
                        itertools.compress(range(len(ischanged)), ischanged)
                    )
                    unchanged = list(
                        itertools.compress(
                            range(len(ischanged)), 1 - np.array(ischanged)
                        )
                    )

                    for i in changed:
                        heappush(self.min_dists, (-new_dists[i], check_list[i][1]))
                        self.vcells[check_list[i][1]] = self.inds[next_ind]
                    for i in unchanged:
                        heappush(self.min_dists, check_list[i])
                else:
                    logger.info("Hopper exhausted!")

            # Store Hausdorff and time information
            if len(self.min_dists) < 1:
                self.r = 0
            else:
                self.r = -self.min_dists[0][0]
            self.rs.append(self.r)
            self.times.append(self.times[-1] + time() - t0)

        return self.path

    def get_wts(self) -> List[int]:
        """Compute the number of points represented by each sketch point.

        Returns
        -------
        list of int
            Weight (Voronoi cell size) for each point in the sketch.
        """
        counter = Counter(self.vcells)
        self.wts = [counter[x] for x in self.path_inds]
        return self.wts

    def get_vdict(self) -> dict:
        """Compute dictionary mapping sketch point indices to their Voronoi cells.

        Returns
        -------
        dict
            Maps each sketch point's external index to a list of member indices.
        """
        result = {}
        for i, c in enumerate(self.vcells):
            if c not in result:
                result[c] = [self.inds[i]]
            else:
                result[c].append(self.inds[i])
        self.vdict = result
        return result

    def __lt__(self, other):
        return self.r > other.r

    def __gt__(self, other):
        return self.r < other.r

    def write(self, filename: str) -> None:
        """Serialize sketch results to a pickle file."""
        data = {
            "path": self.path,
            "vcells": self.vcells,
            "path_inds": self.path_inds,
            "times": self.times,
            "rs": self.rs,
            "wts": self.wts,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def read(self, filename: str) -> None:
        """Load sketch results from a pickle file."""
        with open(filename, "rb") as f:
            hdata = pickle.load(f)
            self.path = hdata["path"]
            self.vcells = hdata["vcells"]
            self.path_inds = hdata["path_inds"]
            self.times = hdata["times"]
            self.rs = hdata["rs"]
            if "wts" in hdata:
                self.wts = hdata["wts"]

    def __getitem__(self, key):
        """Subset the sketch and its associated Voronoi cells."""
        if self.vdict is None:
            self.get_vdict()
        result = deepcopy(self)
        result.path = np.array(self.path)[key]
        result.path_inds = np.array(self.path_inds)[key]
        result.vdict = {c: self.vdict[c] for c in result.path_inds}

        included = np.array([False] * self.numObs)
        for k in result.vdict:
            included[result.vdict[k]] = [True] * len(result.vdict[k])

        where_included = list(
            itertools.compress(list(range(len(self.inds))), included)
        )
        result.inds = np.array(self.inds)[where_included]

        if self.data is None:
            result.data = None
        else:
            result.data = self.data[where_included, :]
        return result

    def compress(self, data: np.ndarray) -> np.ndarray:
        """Return the rows of ``data`` corresponding to sketch points."""
        return data[self.path_inds, :]

    def expand(self, fulldata: np.ndarray, attrs=None):
        """Expand the sketch back to the full dataset via Voronoi cells.

        Parameters
        ----------
        fulldata : np.ndarray
            The full dataset to expand into.
        attrs : array-like, optional
            Attributes to propagate from sketch points to their Voronoi cells.

        Returns
        -------
        np.ndarray or dict
            If ``attrs`` is None, returns the expanded data matrix.
            Otherwise, returns a dict with keys ``'data'`` and ``'attrs'``.
        """
        if self.vdict is None:
            self.get_vdict()

        if attrs is None:
            inds = []
            for c in self.vdict.keys():
                inds += self.vdict[c]
            return fulldata[sorted(inds), :]
        else:
            attrs = np.array(attrs)
            if len(attrs.shape) == 1:
                attrs = attrs.reshape((attrs.shape[0], 1))

            inds = []
            for i in range(attrs.shape[0]):
                cell = self.path_inds[i]
                inds += [(v, attrs[i, :]) for v in self.vdict[cell]]

            inds = sorted(inds)
            subsample = [x[0] for x in inds]
            attrs = np.array([x[1] for x in inds])
            return {"data": fulldata[subsample, :], "attrs": attrs}


class treehopper(hopper):
    """Scalable subsampling by hopping within pre-partitioned data.

    Pre-partitions the data into smaller pieces and runs a ``hopper`` in each
    partition, selecting globally by always hopping in the partition with the
    largest covering radius.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (n_samples, n_features).
    splits : float
        Maximum hops per partition before splitting further (default ``inf``).
    metric : callable
        Distance function.
    inds : sequence of int, optional
        External index labels.
    partition : callable or list of list of int, optional
        Partition function (e.g. ``PCATreePartition``) or explicit partition.
    max_partition_size : int
        Maximum partition size passed to the partition function.
    """

    def __init__(
        self,
        data: np.ndarray,
        splits: float = float("inf"),
        metric=euclidean,
        inds: Optional[Sequence[int]] = None,
        partition: Union[Callable, List[List[int]], None] = None,
        max_partition_size: int = 1000,
    ):
        t0 = time()
        self.times = []
        self.data = data
        self.numObs, self.numFeatures = data.shape

        self.r = float("inf")
        self.rs = []
        if inds is None:
            inds = range(self.numObs)

        self.inds = inds
        self.data = data
        self.path = []
        self.path_inds = []

        self.min_dists = [float("inf")] * self.numObs
        self.avail_inds = list(range(self.numObs))

        self.distfunc = metric
        self.vcells = None
        self.vdict = None
        self.hheap = []
        self.splits = splits
        self.new = True

        if partition is not None:
            if callable(partition):
                logger.info("Pre-partitioning...")
                P = partition(data, max_partition_size, inds)
            else:
                P = partition
            for rows in P:
                h = hopper(data[rows, :], metric, inds=rows)
                h.hop()
                heappush(self.hheap, h)
            logger.info(f"Pre-partitioning done, added {len(self.path)} points")

        self.init_time = time() - t0
        self.times = [self.init_time]

    def hop(self, n_hops: int = 1, store_vcells: bool = True) -> None:
        """Add points to the sketch from the partition with largest radius.

        Parameters
        ----------
        n_hops : int
            Number of points to add.
        store_vcells : bool
            Unused; kept for API compatibility with ``hopper.hop()``.
        """
        for _ in itertools.repeat(None, n_hops):
            t0 = time()
            logger.debug(f"Path length: {len(self.path)}")

            if len(self.hheap) == 0:
                logger.info("Heap starting")
                heappush(
                    self.hheap,
                    hopper(self.data, self.distfunc, range(self.numObs)),
                )

            h = heappop(self.hheap)
            logger.debug(
                f"Hopping with {h.numObs} points, radius {h.r}"
            )

            if h.new:
                h.new = False
                selected = h.path_inds[-1]
                self.path.append(selected)
                self.path_inds.append(self.inds[selected])
                self.r = h.r
                self.rs.append(self.r)
                heappush(self.hheap, h)
                self.times.append(self.times[-1] + time() - t0)
                continue

            self.r = h.r
            self.rs.append(self.r)
            h.hop()

            selected = h.path_inds[-1]
            self.path.append(selected)
            self.path_inds.append(self.inds[selected])

            if len(h.min_dists) > 0:
                if len(h.path) < self.splits:
                    heappush(self.hheap, h)
                else:
                    logger.debug("Splitting partition")
                    h.get_vdict()
                    for vcell in h.vdict.keys():
                        vcelldata = self.data[h.vdict[vcell], :]
                        cell_inds = h.vdict[vcell]

                        newhopper = hopper(
                            vcelldata, metric=self.distfunc, inds=cell_inds, root=0
                        )
                        newhopper.hop()

                        if len(newhopper.min_dists) > 0:
                            heappush(self.hheap, newhopper)
            else:
                logger.info("Hopper exhausted!")

            self.times.append(self.times[-1] + time() - t0)

    def get_vdict(self) -> dict:
        """Collect Voronoi dictionaries from all sub-hoppers.

        Returns
        -------
        dict
            Combined mapping from sketch point indices to Voronoi cell members.
        """
        result = {}
        for h in self.hheap:
            if not h.new:
                result.update(h.get_vdict())
        self.vdict = result
        return result

    def get_vcells(self) -> list:
        """Compute per-point Voronoi cell assignments.

        Returns
        -------
        list of int
            For each point in the dataset, the index of its nearest sketch point.
        """
        result = [0] * self.numObs
        d = self.get_vdict()
        for k in d.keys():
            for v in d[k]:
                result[v] = k
        self.vcells = result
        return result

    def write(self, filename: str) -> None:
        """Serialize sketch results to a pickle file."""
        if self.vcells is None:
            self.get_vcells()
        if self.vdict is None:
            self.get_vdict()
        data = {
            "path": self.path,
            "vcells": self.vcells,
            "path_inds": self.path_inds,
            "vdict": self.vdict,
            "times": self.times,
            "rs": self.rs,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def get_wts(self) -> List[int]:
        """Compute weights (Voronoi cell sizes) for each sketch point.

        Returns
        -------
        list of int
            Weight for each point in the sketch.
        """
        if self.vcells is None:
            self.get_vcells()
        counter = Counter(self.vcells)
        self.wts = [counter[x] for x in self.path_inds]
        return self.wts

    def read(self, filename: str) -> None:
        """Load sketch results from a pickle file."""
        with open(filename, "rb") as f:
            hdata = pickle.load(f)
            self.path = hdata["path"]
            self.path_inds = hdata["path_inds"]
            if "vdict" in hdata:
                self.vdict = hdata["vdict"]
            if "vcells" in hdata:
                self.vcells = hdata["vcells"]
            if "wts" in hdata:
                self.wts = hdata["wts"]
            self.times = hdata["times"]
            self.rs = hdata["rs"]
