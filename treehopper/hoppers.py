import numpy as np
import itertools
from functools import total_ordering
from scipy.spatial.distance import euclidean
from heapq import heappush, heappop, heapify, heapreplace
import pickle

@total_ordering
class hopper:
    def __init__(self, data, metric=euclidean, inds=None, start_r=float('inf'), root=None):


        self.r = start_r #current Haus distance
        self.rs = [] # distances from each sampled point to furthest non-sampled point in voronoi cell
        self.numObs, self.numFeatures = data.shape

        if inds is None:
            inds = range(self.numObs)

        self.inds = inds
        self.data = data
        self.path = []
        self.path_inds = []

        min_dists = [float('Inf')] * self.numObs
        self.min_dists = min_dists

        avail_inds = list(range(self.numObs))
        self.avail_inds = avail_inds

        self.distfunc = metric
        self.vcells=None
        self.vdict=None
        self.root = root


    def hop(self, n_hops=1, store_vcells=True):
        '''generate exact far traversal'''

        sampleSize = min([n_hops, self.numObs])

        for _ in itertools.repeat(None, n_hops):

            if len(self.path) == 0:
                print('beginning traversal!')
                if self.root is None:
                    first = np.random.choice(list(range(len(self.avail_inds))))
                else:
                    first = self.root
                first_ind = self.avail_inds[first]
                first_pt = self.data[first_ind,:]


                self.path.append(first_ind)
                self.path_inds.append(self.inds[first_ind])
                del self.avail_inds[first]
                del self.min_dists[first]

                self.min_dists = [min(self.min_dists[pos], self.distfunc(self.data[ind,:],first_pt)) for pos, ind in enumerate(self.avail_inds)]

                self.rs = [max([0]+self.min_dists)]
                self.r = max(self.rs)

                self.closest = [0]*len(self.avail_inds)
                # self.ptrs = [0]

                if store_vcells:
                    self.vcells = [self.inds[first_ind]] * self.numObs
                    self.vdict = {}
                    self.vdict[self.inds[first_ind]] = [self.inds[first_ind]]+[self.inds[x] for x in self.avail_inds]
            else:
                #print(len(self.path))
                next_pos = self.min_dists.index(max(self.min_dists))
                next_ind = self.avail_inds[next_pos]
                next_pt = self.data[next_ind,:]

                self.path.append(next_ind)
                self.path_inds.append(self.inds[next_ind])
                # self.ptrs.append(self.closest[next_pos])

                if store_vcells:
                    #initialize voronoi cell with self
                    #prepare to rebuild voronoi dictionary
                    self.vcells[next_ind]=self.inds[next_ind]
                    self.vdict = {self.path_inds[x]:[self.path_inds[x]] for x in range(len(self.path))}


                #reset rs to acommodate new point!!
                self.rs = [0]*len(self.path)

                #print(len(self.path))
                del self.avail_inds[next_pos]
                del self.closest[next_pos]
                del self.min_dists[next_pos]


                for pos, ind in enumerate(self.avail_inds):
                    cur_dist = self.distfunc(self.data[ind,:],next_pt)
                    if cur_dist < self.min_dists[pos]:
                        self.closest[pos] = len(self.path) - 1
                        self.min_dists[pos] = cur_dist

                        if store_vcells: #todo make this work only once
                            #update voronoi cell with self
                            self.vcells[ind] = self.inds[next_ind]

                    #update rs
                    self.rs[self.closest[pos]] = max(self.rs[self.closest[pos]], self.min_dists[pos])
                    if store_vcells:
                        self.vdict[self.path_inds[self.closest[pos]]].append(self.inds[ind])

                self.r = max(self.rs)
                print(self.r)

        return(self.path)

    def __lt__(self, other):
        return self.r > other.r

    def __gt__(self, other):
        return self.r < other.r

    def write(self, filename):
        data = {'path':self.path, 'vcells':self.vcells, 'path_inds':self.path_inds}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def read(self, filename):
        '''load hopData file and store into its values'''
        with open(filename, 'rb') as f:
            hdata = pickle.load(f)

            self.path = hdata['path']
            self.vcells = hdata['vcells']
            self.path_inds = hdata['path_inds']

class treehopper:
    def __init__(self, data, splits=2, metric=euclidean, inds=None):
        self.data = data
        self.numObs, self.numFeatures = data.shape

        if inds is None:
            inds = range(self.numObs)

        self.inds = inds
        self.data = data
        self.path = []
        self.path_inds = []

        min_dists = [float('Inf')] * self.numObs
        self.min_dists = min_dists

        avail_inds = list(range(self.numObs))
        self.avail_inds = avail_inds

        self.distfunc = metric
        self.vcells=None
        self.vdict=None
        self.hheap = []
        self.splits = splits


    def hop(self, n_hops=1, store_vcells=True):

        for _ in itertools.repeat(None, n_hops):
            print(len(self.path))
            if len(self.hheap) == 0: #start heaping
                print('heap starting')
                heappush(self.hheap, hopper(self.data,self.distfunc, range(self.numObs)))

            h = heappop(self.hheap)
            print('radius {}'.format(h.r))

            if len(h.avail_inds) < 1: #hopper exhausted, can't hop anymore
                continue

            h.hop() #add furthest point to h, append it to path
            next = h.path_inds[-1]

            self.path.append(next)
            self.path_inds.append(self.inds[next])

            if len(h.path) < self.splits:
                heappush(self.hheap, h)
            else:
                print('splitting')
                #split into sub-hoppers

                for vcell in h.vdict.keys():
                    vcelldata = self.data[h.vdict[vcell],:]

                    #rad = h.rs[h.path_inds.index(vcell)]

                    inds = h.vdict[vcell]
                    #print(sorted(inds))

                    newhopper = hopper(vcelldata, metric=self.distfunc, inds=inds, root=0)
                    newhopper.hop() #Initializes r, stops sampling the root
                    #print(newhopper.vdict.keys())
                    heappush(self.hheap,newhopper)

    def get_vdict(self):
        result = {}
        for h in self.hheap:
            result.update(h.vdict)
        return(result)

    def get_vcells(self):
        result = [0]*self.numObs
        d = self.get_vdict()
        for k in d.keys():
            for v in d[k]:
                result[v] = k
        #print(self.get_vdict().keys())

        return(result)
