# Hopper: Structure-preserving subsampling

Hopper implements the **greedy k-centers algorithm**, iteratively generating a farthest-first traversal of the input data. The resulting subset realizes a 2-approximation to the optimal Hausdorff distance between the subset and the full dataset.

> DeMeo B, Berger B. Hopper: a mathematically optimal algorithm for sketching biological data. *Bioinformatics*. 2020 Jul 1;36(Suppl_1):i236-i241. [doi:10.1093/bioinformatics/btaa408](https://doi.org/10.1093/bioinformatics/btaa408)

## Installation

```bash
pip install .
```

Or install in development mode:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from treehopper import hopper

# X is your data matrix: (n_samples, n_features)
h = hopper(X)
sketch = h.hop(1000)       # indices of 1000 sketch points
X_small = X[sketch, :]     # the subsampled data
```

Additional points can be added incrementally by calling `h.hop()` again. The full sketch is stored in `h.path`.

### Attributes

| Attribute | Description |
|-----------|-------------|
| `h.path` | Row indices of sketch points (traversal order) |
| `h.path_inds` | External index labels of sketch points |
| `h.rs` | Covering radius after each hop |
| `h.times` | Cumulative wall-clock time after each hop |
| `h.vcells` | Voronoi cell assignment for every input point |

Call `h.get_vdict()` to obtain a dictionary mapping each sketch point index to the indices of its Voronoi cell members.

## Treehopper for larger datasets

`hopper.hop()` takes time proportional to the dataset size per hop. For large datasets, `treehopper` reduces runtime by pre-partitioning the data and running a `hopper` within each partition:

```python
from treehopper import treehopper, PCATreePartition

th = treehopper(X, partition=PCATreePartition, max_partition_size=500)
sketch = th.hop(10000)
```

The `max_partition_size` parameter controls the time-performance tradeoff (runtime scales linearly with it).

**Note:** `treehopper` seeds its sketch by drawing a point from each partition, so the sketch may contain slightly more than the requested number of points. To get exactly *k* points:

```python
sketch = th.path[:k]
```

All `hopper` methods and attributes are available on `treehopper`. The `vcells` and `rs` attributes are upper bounds, since `treehopper` does not compare points across partitions.

## Evaluating sketch quality

```python
from treehopper import haus

# Hausdorff distance from the full data to the sketch
h_dist = haus(X, sketch)
```

## Citation

```bibtex
@article{demeo2020hopper,
  title={Hopper: a mathematically optimal algorithm for sketching biological data},
  author={DeMeo, Benjamin and Berger, Bonnie},
  journal={Bioinformatics},
  volume={36},
  number={Suppl\_1},
  pages={i236--i241},
  year={2020},
  doi={10.1093/bioinformatics/btaa408},
  pmid={32657375}
}
```

## License

[MIT](LICENSE)
