# Hopper: Diverse sampling for biological datasets

Hopper implements the greedy k-centers algorithm, iteratively generating a farthest-first traversal of the input data. The resulting subset realizes a 2-approximation to the optimal Hausdorff distance between the subset and the full dataset. See the full pre-print here: https://www.biorxiv.org/content/10.1101/835033v1


## Usage
To sketch a dataset, import the hopper class, and first pass the dataset into the Hopper constructor: 
```
from hoppers import hopper # class is defined in treehoppers/hoppers.py

h = hopper(X) # X is the input data, with one row per observation and one column per feature
```

The `hop()` method adds a point to the sketch, and returns the entire sketch generated thus far. To produce a sketch of size _k_=1000, and use it to subset the data, you can run
```
k=1000
sketch = h.hop(k)
X_small = X[sketch,:]
```
Additional points can be added to the sketch by running `h.hop()` again with the same hopper object. The full sketch generated thus far is stored in the attribute `h.path`. In addition, Hopper keeps track of the total computation time used for sketching in the attribute `times`, and the covering radius in the attribute `rs`. While generating sketches, Hopper maintains the Voronoi partition of the full dataset, i.e. the set of points closest to each subsampled point. The attribute `vcells` stores a 1-D array listing, for each index into `X`, the index of the subsampled point that it is closest to. To obtain a dictionary mapping each subsampled index to the indices of its Voronoi cell members, run `h.get_vdict()`, which returns the dictionary and stores it in the attribute `vdict`.
