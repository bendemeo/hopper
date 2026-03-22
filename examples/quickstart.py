"""Quick-start example: subsample random data with hopper and treehopper."""

import numpy as np
from treehopper import hopper, treehopper, PCATreePartition
from treehopper.utils import haus

rng = np.random.RandomState(42)
X = rng.normal(size=(2000, 10))

# --- Basic hopper ---
print("=== Hopper ===")
h = hopper(X, root=0)
h.hop(100)
print(f"Sketch size: {len(h.path)}")
print(f"Hausdorff distance: {haus(X, h.path):.4f}")
print(f"Covering radii (first 5): {[f'{r:.3f}' for r in h.rs[:5]]}")

# Voronoi cell weights
h.get_wts()
print(f"Total weight (should equal n): {sum(h.wts)}")

# --- Treehopper for larger data ---
print("\n=== Treehopper ===")
th = treehopper(X, partition=PCATreePartition, max_partition_size=200)
th.hop(100)
print(f"Sketch size: {len(th.path)}")
print(f"Hausdorff distance: {haus(X, th.path[:100]):.4f}")

# --- Compress and expand ---
print("\n=== Compress / Expand ===")
X_small = h.compress(X)
print(f"Compressed shape: {X_small.shape}")

X_expanded = h.expand(X)
print(f"Expanded shape: {X_expanded.shape}")
