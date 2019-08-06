from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean


def haus(data, sample):
    dist = pairwise_distances(data[sample, :], data, n_jobs=-1)
    return(dist.min(0).max())


def haus_curve(data, ordering, distfunc=euclidean, max_len=5000):
    result = []
    cur_haus = float('Inf')
    min_dists = float('Inf')*data.shape[0]
    for i in range(len(ordering)):
        if i > max_len:
            break
        new = data[ordering[i]].tolist()
        new_dists = pairwise_distances([new], data).flatten()
        min_dists = np.minimum(min_dists, new_dists)
        cur_haus = max(min_dists)
        result.append(cur_haus)

    return(result)
