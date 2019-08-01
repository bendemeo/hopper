import numpy as np
import itertools
from scipy.spatial.distance import euclidean


class hopper:
    def __init__(self, data, metric=euclidean):
        self.numObs, self.numFeatures = data.shape

        self.data = data
        self.path = []

        min_dists = [float('Inf')] * self.numObs
        self.min_dists = min_dists

        avail_inds = list(range(self.numObs))
        self.avail_inds = avail_inds

        self.distfunc = metric
        self.vcells=None


    def hop(self, n_hops=1, store_vcells=True):
        '''generate exact far traversal'''

        sampleSize = min([n_hops, self.numObs])

        for _ in itertools.repeat(None, n_hops):

            if len(self.path) == 0:
                print('beginning traversal!')
                first = np.random.choice(list(range(len(self.avail_inds))))
                first_ind = self.avail_inds[first]
                first_pt = self.data[first_ind,:]


                self.path.append(first_ind)
                del self.avail_inds[first]
                del self.min_dists[first]

                self.min_dists = [min(self.min_dists[pos], self.distfunc(self.data[ind,:],first_pt)) for pos, ind in enumerate(self.avail_inds)]

                self.closest = [0]*len(self.avail_inds)
                # self.ptrs = [0]

                if store_vcells:
                    self.vcells = [0] * self.data.shape[0]
            else:
                next_pos = self.min_dists.index(max(self.min_dists))
                next_ind = self.avail_inds[next_pos]
                next_pt = self.data[next_ind,:]

                self.path.append(next_ind)
                # self.ptrs.append(self.closest[next_pos])

                #initialize voronoi cell with self
                self.vcells[next_ind]= len(self.path)-1


                print(len(self.path))
                del self.avail_inds[next_pos]
                del self.closest[next_pos]
                del self.min_dists[next_pos]


                for pos, ind in enumerate(self.avail_inds):
                    cur_dist = self.distfunc(self.data[ind,:],next_pt)
                    if cur_dist < self.min_dists[pos]:
                        self.closest[pos] = len(self.path) - 1
                        self.min_dists[pos] = cur_dist

                        if store_vcells:
                            #update voronoi cell with self
                            self.vcells[ind] = len(self.path) - 1

        return(self.path)
