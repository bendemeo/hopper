import numpy as np
import itertools
from functools import total_ordering
from scipy.spatial.distance import euclidean
from heapq import heappush, heappop, heapify, heapreplace
from sklearn.metrics import pairwise_distances
import pickle



def RPartition(data, max_partition_size=1000, inds=None):
    '''Median-partition along dimensions until each partition is small'''
    partitions = []
    if inds is None:
        inds = list(range(data.shape[0]))

    #initialize to trivial partition
    #tuple of (-size, next partitioning dimension, rows in partition)
    heappush(partitions, (-1*data.shape[0], 0, list(range(data.shape[0]))))

    current_partition = heappop(partitions)
    while(len(current_partition[2]) > max_partition_size):
        dim = current_partition[1]
        rows = current_partition[2]
        vals = data[rows, dim].tolist()

        mid = np.median(vals)
        split = vals > mid

        p1 = list(itertools.compress(rows, split))
        p2 = list(itertools.compress(rows, 1-split))


        newdim = (dim + 1) % data.shape[1] #cycle dimensions

        heappush(partitions, (-1*len(p1), newdim, p1))
        heappush(partitions, (-1*len(p2), newdim, p2))

        current_partition = heappop(partitions)

    heappush(partitions, current_partition)
    return([x[2] for x in partitions])







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
                print('beginning traversal! {} items to traverse'.format(self.numObs))
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

                self.max_pos = self.min_dists.index(max(self.min_dists))
                self.r = self.min_dists[self.max_pos]

                self.closest = [0]*len(self.avail_inds)
                # self.ptrs = [0]

                if store_vcells:
                    self.vcells = [self.inds[first_ind]] * self.numObs
                    self.vdict = {}
                    self.vdict[self.inds[first_ind]] = [self.inds[first_ind]]+[self.inds[x] for x in self.avail_inds]
            else:

                #print(len(self.path))
                next_pos = self.max_pos
                next_ind = self.avail_inds[next_pos]
                next_pt = self.data[next_ind,:].reshape((1,self.numFeatures))

                self.path.append(next_ind)
                self.path_inds.append(self.inds[next_ind])
                # self.ptrs.append(self.closest[next_pos])

                if store_vcells:
                    #initialize voronoi cell with self
                    #prepare to rebuild voronoi dictionary
                    self.vcells[next_ind]=self.inds[next_ind]
                    #self.vdict = {self.path_inds[x]:[self.path_inds[x]] for x in range(len(self.path))}

                #reset rs to acommodate new point!!
                self.rs = [0]*len(self.path)

                #print(len(self.path))
                del self.avail_inds[next_pos]
                del self.closest[next_pos]
                del self.min_dists[next_pos]


                #places where dists MAY be changed
                check = np.array(self.min_dists) > float(self.r)/2
                check_pos = list(itertools.compress(range(len(self.avail_inds)),check))
                check_inds = np.array(self.avail_inds)[check_pos]

                #compute distances
                dists = pairwise_distances(np.array(next_pt), self.data[check_inds,:])[0,:]
                dists = np.array(dists)

                #find places where distance WILL be changed
                prev_dists = np.array(self.min_dists)[check_pos]
                change = (dists < prev_dists)

                change_pos = list(itertools.compress(range(len(check_pos)),change))
                change_inds = np.array(check_pos)[change_pos]

                #update minimum distances and vcells, if applicable
                for i, idx in enumerate(change_pos):
                    #print(self.min_dists[idx])
                    self.min_dists[change_inds[i]] = dists[idx]
                    self.closest[change_inds[i]] = len(self.path) - 1
                    # self.r = max(self.r, )
                    # self.rs[len(self.path)-1] = max(self.rs[len(self.path)-1],
                    #                                 self.min_dists[idx])
                    if store_vcells:
                        self.vcells[check_inds[change_pos[i]]] = self.inds[next_ind]


                #This line may be a bit slow -- consider sorting min_dists?
                self.max_pos = self.min_dists.index(max(self.min_dists))
                self.r = self.min_dists[self.max_pos]

                # #TODO speed up with sklearn.metrics.pairwise_distances (I did it! Hence this is commented)
                # for pos, ind in enumerate(self.avail_inds):
                #
                #     if self.min_dists[pos] < float(self.r)/2.:
                #         if store_vcells:
                #             self.vdict[self.path_inds[self.closest[pos]]].append(self.inds[ind])
                #         continue
                #         # no need to check; old point is closer
                #
                #     cur_dist = self.distfunc(self.data[ind,:],next_pt)
                #     if cur_dist < self.min_dists[pos]:
                #         self.closest[pos] = len(self.path) - 1
                #         self.min_dists[pos] = cur_dist
                #
                #         if store_vcells: #todo make this work only once
                #             #update voronoi cell with self
                #             self.vcells[ind] = self.inds[next_ind]
                #
                #     #update rs
                #     self.rs[self.closest[pos]] = max(self.rs[self.closest[pos]], self.min_dists[pos])
                #
                #     if store_vcells:
                #         self.vdict[self.path_inds[self.closest[pos]]].append(self.inds[ind])
                #
                # self.r = max(self.rs)
                # print('r values: {}'.format(self.rs))
                #print(self.r)

        return(self.path)

    def get_vdict(self):
        #compute dictionary from cell ids
        result = {}
        for i,c in enumerate(self.vcells):
            if c not in result:
                result[c] = [self.inds[i]]
            else:
                result[c].append(self.inds[i])
        return(result)

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
    def __init__(self, data, splits=2, metric=euclidean, inds=None,
                 pre_partition = False,
                 max_partition_size=1000):

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

        if pre_partition:
            '''use RP-partition to split data beforehand'''
            print('Pre-partitioning...')
            P = RPartition(data, max_partition_size, inds)
            for rows in P:
                h = hopper(data[rows,:], metric, rows)
                h.hop() #hop once to set root
                next = h.path_inds[-1]
                self.path.append(next)
                self.path_inds.append(inds[next])
                heappush(self.hheap, h)
            print('Pre-partitioning done, added {} points'.format(len(self.path)))






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

            print('hopping with {} points'.format(h.numObs))
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

                    avail_idx = np.array(h.inds)[h.avail_inds].tolist()



                    #mindists = [0]+[h.min_dists[avail_idx.index(x)] for x in h.vdict[vcell][1:]]
                    #rad = h.rs[h.path_inds.index(vcell)]

                    inds = h.vdict[vcell]
                    #print(sorted(inds))

                    newhopper = hopper(vcelldata, metric=self.distfunc, inds=inds, root=0)

                    #newhopper.min_dists = mindists

                    newhopper.hop() #Initializes r, stops sampling the root
                    #print(newhopper.vdict.keys())
                    heappush(self.hheap,newhopper)

    def get_vdict(self):
        result = {}
        for h in self.hheap:
            result.update(h.get_vdict())

        self.vdict = result
        return(result)

    def get_vcells(self):
        result = [0]*self.numObs
        d = self.get_vdict()
        for k in d.keys():
            for v in d[k]:
                result[v] = k
        #print(self.get_vdict().keys())
        self.vcells = result

        return(result)

    def write(self, filename):
        if(self.vcells is None):
            self.get_vcells()
        if self.vdict is None:
            self.get_vdict()
        data = {'path':self.path, 'vcells':self.vcells, 'path_inds':self.path_inds,
                'vdict':self.vdict}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def read(self, filename):
        '''load hopData file and store into its values'''
        with open(filename, 'rb') as f:
            hdata = pickle.load(f)

            self.path = hdata['path']
            self.path_inds = hdata['path_inds']
            if 'vdict' in hdata:
                self.vdict = hdata['vdict']
            if 'vcells' in hdata:
                self.vcells = hdata['vcells']
