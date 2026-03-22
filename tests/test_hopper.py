"""Tests for hopper and treehopper classes."""

import numpy as np
import pytest

from treehopper import hopper, treehopper, PCATreePartition, RPartition
from treehopper.utils import haus


@pytest.fixture
def gaussian_data():
    rng = np.random.RandomState(42)
    return rng.normal(size=(500, 3))


class TestHopper:
    def test_basic_hop(self, gaussian_data):
        h = hopper(gaussian_data)
        path = h.hop(10)
        assert len(path) == 10
        assert len(set(path)) == 10  # all unique

    def test_incremental_hop(self, gaussian_data):
        h = hopper(gaussian_data)
        h.hop(5)
        h.hop(5)
        assert len(h.path) == 10

    def test_path_inds_match(self, gaussian_data):
        h = hopper(gaussian_data)
        h.hop(10)
        assert len(h.path_inds) == len(h.path)

    def test_rs_decreasing(self, gaussian_data):
        h = hopper(gaussian_data)
        h.hop(20)
        # After the first point, radii should be non-increasing
        for i in range(2, len(h.rs)):
            assert h.rs[i] <= h.rs[i - 1] + 1e-10

    def test_vcells_assigned(self, gaussian_data):
        h = hopper(gaussian_data)
        h.hop(10)
        assert h.vcells is not None
        assert len(h.vcells) == gaussian_data.shape[0]

    def test_vdict(self, gaussian_data):
        h = hopper(gaussian_data)
        h.hop(10)
        vdict = h.get_vdict()
        total_members = sum(len(v) for v in vdict.values())
        assert total_members == gaussian_data.shape[0]

    def test_weights_sum(self, gaussian_data):
        h = hopper(gaussian_data)
        h.hop(10)
        h.get_wts()
        assert sum(h.wts) == gaussian_data.shape[0]

    def test_compress(self, gaussian_data):
        h = hopper(gaussian_data)
        h.hop(10)
        compressed = h.compress(gaussian_data)
        assert compressed.shape == (10, 3)

    def test_hausdorff_decreases(self, gaussian_data):
        h = hopper(gaussian_data)
        h.hop(20)
        h1 = haus(gaussian_data, h.path[:5])
        h2 = haus(gaussian_data, h.path[:20])
        assert h2 < h1

    def test_deterministic_with_root(self, gaussian_data):
        h1 = hopper(gaussian_data, root=0)
        h1.hop(10)
        h2 = hopper(gaussian_data, root=0)
        h2.hop(10)
        assert h1.path == h2.path

    def test_no_data_raises(self):
        with pytest.raises((ValueError, TypeError)):
            h = hopper(None)
            h.hop()

    def test_custom_metric(self, gaussian_data):
        from scipy.spatial.distance import cosine
        h = hopper(gaussian_data, metric=cosine, root=0)
        h.hop(10)
        assert len(h.path) == 10
        # Verify radii are computed with cosine distance, not euclidean
        h2 = hopper(gaussian_data, root=0)
        h2.hop(10)
        assert h.rs != h2.rs


class TestTreehopper:
    def test_basic_hop(self, gaussian_data):
        th = treehopper(
            gaussian_data, partition=PCATreePartition, max_partition_size=100
        )
        th.hop(20)
        assert len(th.path) >= 20

    def test_rpartition(self, gaussian_data):
        th = treehopper(
            gaussian_data, partition=RPartition, max_partition_size=100
        )
        th.hop(20)
        assert len(th.path) >= 20

    def test_vcells(self, gaussian_data):
        th = treehopper(
            gaussian_data, partition=PCATreePartition, max_partition_size=100
        )
        th.hop(20)
        vcells = th.get_vcells()
        assert len(vcells) == gaussian_data.shape[0]

    def test_vdict(self, gaussian_data):
        th = treehopper(
            gaussian_data, partition=PCATreePartition, max_partition_size=100
        )
        th.hop(20)
        vdict = th.get_vdict()
        assert len(vdict) > 0

    def test_weights(self, gaussian_data):
        th = treehopper(
            gaussian_data, partition=PCATreePartition, max_partition_size=100
        )
        th.hop(50)
        th.get_wts()
        assert len(th.wts) == len(th.path)
        assert all(w >= 1 for w in th.wts)


class TestPartitions:
    def test_pca_partition_sizes(self, gaussian_data):
        partitions = PCATreePartition(gaussian_data, max_partition_size=100)
        for p in partitions:
            assert len(p) <= 100
        all_indices = sorted(sum(partitions, []))
        assert all_indices == list(range(gaussian_data.shape[0]))

    def test_r_partition_sizes(self, gaussian_data):
        partitions = RPartition(gaussian_data, max_partition_size=100)
        for p in partitions:
            assert len(p) <= 100
        all_indices = sorted(sum(partitions, []))
        assert all_indices == list(range(gaussian_data.shape[0]))
