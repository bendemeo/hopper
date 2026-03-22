from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from scipy.spatial.distance import euclidean
import numpy as np


def haus(data, sample, max_mem=float('inf')):
    if max_mem == float('inf'):
        dist = pairwise_distances(data[sample, :], data, n_jobs=-1)
        return(dist.min(0).max())
    else:

        dists = pairwise_distances_chunked(data, data[sample,:],
                                           reduce_func = lambda x,y: x.min(1),
                                           working_memory = max_mem)

        h = 0
        for x in dists:
            h = max([h, max(x)])
            print(h)
        return h



def haus_curve(data, ordering, distfunc=euclidean, max_len=5000):

    print('starting')
    result = []
    cur_haus = float('Inf')
    min_dists = float('Inf')*data.shape[0]
    for i in range(len(ordering)):
        print(i)
        if i > max_len:
            break
        new = data[ordering[i],:]
        new_dists = pairwise_distances(new, data).flatten()
        min_dists = np.minimum(min_dists, new_dists)
        cur_haus = max(min_dists)
        result.append(cur_haus)

    return(result)
