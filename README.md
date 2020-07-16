# Hopper: Diverse sampling for biological datasets

Hopper implements the greedy k-centers algorithm, iteratively generating a farthest-first traversal of the input data. The resulting subset realizes a 2-approximation to the optimal Hausdorff distance between the subset and the full dataset. See the full pre-print here: https://www.biorxiv.org/content/10.1101/835033v1


## Usage
To sketch a dataset, import the hopper class, and first pass the dataset into the Hopper constructor: 
```
import hopper 

h = hopper.hopper(X) # X is the input data, with one row per observation and one column per feature
```

The `hop()` method adds a point to the sketch, and returns the entire sketch generated thus far. To produce a sketch of size _k_=1000, and use it to subset the data, you can run
```
k=1000
sketch = h.hop(k)
X_small = X[sketch,:]
```
Additional points can be added to the sketch by running `h.hop()` again with the same hopper object. The full sketch generated thus far is stored in the attribute `h.path`. In addition, Hopper keeps track of the total computation time used for sketching in the attribute `times`, and the covering radius in the attribute `rs`. 

While generating sketches, Hopper maintains the Voronoi partition of the full dataset, i.e. the set of points closest to each subsampled point. The attribute `vcells` stores a 1-D array listing, for each index into `X`, the index of the subsampled point that it is closest to. To obtain a dictionary mapping each subsampled index to the indices of its Voronoi cell members, run `h.get_vdict()`, which returns the dictionary and stores it in the attribute `vdict`.

### Using Treehopper for larger datasets

The function `hopper.hop()` takes time proportional to the size of the input data. This makes it difficult to generate big sketches of big datasets. The class `treehopper` reduces this runtime drastically with minimal degradation in performance. It does so by pre-partitioning the data into smaller pieces, and instantiating a `hopper` object in each. 

Treehopper objects accept arbitrary partitions, which are passed either as explicit lists of indices or as callable functions that generate these lists from the original data. To get started, I recommend using the `PCATreePartition` function implemented in `hoppers.py`. You can also specify the maximum size of a partition (default 1000), which controls the time-performance tradeoff. The runtime scales linearly with the partition size. Other parameters can be ignored for now. 

Here is an example script:
```
import hopper


#Load some dataset X here

th = treehopper(X, partition=PCATreePartition, max_partition_size=500)
sketch = th.hop(10000)
```
As before, In the current implementation, `treehopper` seeds its sketch by drawing a point from each partition at random. As a result, in the above example, `sketch` will contain more than 10000 points after the script runs. This is counterintuitive, and may be remedied in future versions; in the meantime, if you want an exact number of points, simply remove points from the end of the output:

 ```
 # I want exactly 10000 points!
 sketch = sketch[:10000] #or sketch = th.path[:10000]
 ```
 
All other methods and attributes are identical to those of `hopper`. Although the exact Voronoi cells are not computed, the attribute `th.vcells` maps each point of `X` to the nearest subsampled point in its starting partition, and you can call `th.get_vdict()` to produce a dictionary in the same way. Similarly, the values stored in the `rs` attribute may be larger than the actual covering radius, because `treehopper` does not compare points between partitions. 





