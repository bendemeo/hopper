import numpy as np
import itertools
from functools import total_ordering
from scipy.spatial.distance import euclidean
from heapq import heappush, heappop, heapify, heapreplace
from sklearn.metrics import pairwise_distances
import pickle
from time import time
from fbpca import pca
from contextlib import suppress
from copy import deepcopy
from collections import Counter



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




def PCATreePartition(data, max_partition_size=1000, inds=None):
    partitions = []
    if inds is None:
        inds = list(range(data.shape[0]))

    heappush(partitions, (-1*data.shape[0], list(range(data.shape[0]))))

    current_partition = heappop(partitions)

    while(len(current_partition[1]) > max_partition_size):
        rows = current_partition[1]
        subset = data[rows,:]

        U,s,Vt = pca(subset, k=1)

        pcvals = U[:,:1] * s[:1]
        #print(pcvals)

        mid = np.median(pcvals)
        split = pcvals > mid

        p1 = list(itertools.compress(rows, split))
        p2 = list(itertools.compress(rows, 1-split))

        heappush(partitions, (-1*len(p1), p1))
        heappush(partitions, (-1*len(p2),  p2))

        current_partition = heappop(partitions)

    heappush(partitions, current_partition)
    return([x[1] for x in partitions])


@total_ordering
class hopper:
    def __init__(self, data, metric=euclidean, inds=None, start_r=float('inf'), root=None):

        t0 = time()
        self.times = [] # store runtimes after each hop
        self.r = start_r #current Haus distance
        self.rs = [] # keep track of haus after each time point

        if data is None:
            self.numObs = None
            self.numFeatures = None
        else:
            self.numObs, self.numFeatures = data.shape

        if inds is None:
            inds = range(self.numObs)

        self.inds = inds
        self.data = data
        self.path = [] #init empty traversal
        self.path_inds = []

        self.min_dists = [] #dist to closest pt in traversal

        # avail_inds = list(range(self.numObs))
        # self.avail_inds = avail_inds

        self.distfunc = metric
        self.vcells=None
        self.vdict=None
        self.wts = None

        self.root = root
        self.init_time = time()-t0
        self.times.append(self.init_time)
        self.new = True # for Treehopper

    def hop(self, n_hops=1, store_vcells=True):
        '''generate exact far traversal'''

        if self.data is None: #only stores path info
            raise Exception('no data stored in this hopper!')

        for _ in itertools.repeat(None, n_hops):
            t0 = time()
            print('beginning traversal! {} items to traverse'.format(self.numObs))

            if len(self.path) == 0:
                #set starting point, initialize dist heap
                if self.root is None:
                    first = np.random.choice(list(range(self.numObs)))
                else:
                    first = self.root

                self.path.append(first)
                self.path_inds.append(self.inds[first])

                first_pt = self.data[first,:].reshape((1,self.numFeatures))

                start_dists = pairwise_distances(first_pt, self.data, metric=self.distfunc)[0,:]
                start_dists = np.array(start_dists)

                #initialize min distances heap
                for ind in range(self.numObs):
                    if ind != first:
                        heappush(self.min_dists, (-1*start_dists[ind], ind))

                self.vcells = [self.inds[first]] * self.numObs

            else:
                if len(self.min_dists) < 1:
                    print('hopper exhausted!')
                    break

                next_ind = heappop(self.min_dists)[1]
                next_pt = self.data[next_ind,:].reshape((1,self.numFeatures))

                self.path.append(next_ind)
                self.path_inds.append(self.inds[next_ind])

                if store_vcells:
                    self.vcells[next_ind]=self.inds[next_ind]


                #find places where dists MAY be changed
                check_inds = [] # what indices to check
                check_list = [] # list of heap elements to check
                prev_dists = [] # prior distances

                r = float('inf')

                if len(self.min_dists) > 0:
                    while r > self.r/2 and len(self.min_dists) > 0:
                        curtuple = heappop(self.min_dists)
                        check_inds.append(curtuple[1])
                        check_list.append(curtuple)
                        prev_dists.append(-1*curtuple[0])
                        r = -1*curtuple[0]

                    heappush(self.min_dists, curtuple)

                    print('checking {} points'.format(len(check_list)))

                    #compute pairwise distances
                    new_dists = pairwise_distances(np.array(next_pt), self.data[check_inds,:])[0,:]
                    new_dists = np.array(new_dists)

                    #filter by changed vs. unchanged. Faster than looping
                    ischanged = (new_dists < prev_dists)
                    changed = list(itertools.compress(range(len(ischanged)),ischanged))
                    unchanged = list(itertools.compress(range(len(ischanged)),1-np.array(ischanged)))

                    for i in changed:
                        new = new_dists[i]
                        idx = check_list[i][1]
                        heappush(self.min_dists, (-1*new, idx))
                        self.vcells[idx] = self.inds[next_ind]
                    for i in unchanged:
                        heappush(self.min_dists, check_list[i])
                else:
                    print('hopper exhausted!')


            #store Hausdorff and time information
            if len(self.min_dists) < 1:
                self.r = 0
            else:
                self.r = -1*self.min_dists[0][0]
            self.rs.append(self.r)
            self.times.append(self.times[-1]+time()-t0)


        return(self.path)

    def get_wts(self):
        #See how many points each represents
        counter = Counter(self.vcells)
        self.wts = [counter[x] for x in self.path_inds]

    def get_vdict(self):
        #compute dictionary from cell ids
        result = {}
        for i,c in enumerate(self.vcells):
            if c not in result:
                result[c] = [self.inds[i]]
            else:
                result[c].append(self.inds[i])
        self.vdict = result
        return(result)

    def __lt__(self, other):
        return self.r > other.r

    def __gt__(self, other):
        return self.r < other.r

    def write(self, filename):
        data = {'path':self.path, 'vcells':self.vcells, 'path_inds':self.path_inds,
                'times':self.times,'rs':self.rs, 'wts':self.wts}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def read(self, filename):
        '''load hopData file and store into its values'''
        with open(filename, 'rb') as f:
            hdata = pickle.load(f)

            self.path = hdata['path']
            self.vcells = hdata['vcells']
            self.path_inds = hdata['path_inds']
            self.times = hdata['times']
            self.rs = hdata['rs']
            if 'wts' in hdata:
                self.wts = hdata['wts']

    def __getitem__(self, key):
        #subsets path, and includes from the data only those indices nearest
        #to the subset path points

        if self.vdict is None:
            self.get_vdict()
        result = deepcopy(self)
        result.path = np.array(self.path)[key]
        result.path_inds = np.array(self.path_inds)[key]
        result.vdict = {c:self.vdict[c] for c in result.path_inds}


        #included indices, in sorted order
        included = np.array([False]*self.numObs)

        #find out which points are included
        for k in result.vdict:
            included[result.vdict[k]] = [True]*len(result.vdict[k])

        where_included = list(itertools.compress(list(range(len(self.inds))), included))
        #print(where_included)
        result.inds = np.array(self.inds)[where_included]

        if self.data is None:
            result.data = None
        else:
            result.data = self.data[where_included,:]
        return(result)

    def compress(self, data):
        return data[self.path_inds,:]

    def expand(self, fulldata, attrs=None):
        #use a hopper's data to nearest-neighbor classify full data
        #way to retain attributes of small data e.g. louvain clusters?
        #all attributes stored in the full data are kept!

        if self.vdict is None:
            self.get_vdict()

        inds = []

        if attrs is None:
            for c in self.vdict.keys():
                inds += self.vdict[c]

            #print(inds)
            return fulldata[sorted(inds),:]

        else:
            #ensure it's 2D
            attrs = np.array(attrs)
            if len(attrs.shape) == 1:
                attrs = attrs.reshape((attrs.shape[0], 1))


            inds = []
            for i in range(attrs.shape[0]):
                cell = self.path_inds[i]
                inds += [(v,attrs[i,:]) for v in self.vdict[cell]]

            inds = sorted(inds)
            subsample = [x[0] for x in inds]
            attrs = np.array([x[1] for x in inds])
            result = {'data':fulldata[subsample,:], 'attrs':attrs}
            return(result)

    # def expand_attr(self, attrs, fulldata):
    #     '''Expand a label by nearest-neighbor classification'''
    #
    #     if self.vdict is None:
    #         self.get_vdict()
    #
    #     #ensure it's 2D
    #     attrs = np.array(attrs)
    #     if len(attrs.shape) == 1:
    #         attrs = attrs.reshape((attrs.shape[0], 1))
    #
    #
    #     inds = []
    #     for i in range(attrs.shape[0]):
    #         cell = self.path_inds[i]
    #         inds += [(v,attrs[i,:]) for v in self.vdict[cell]]
    #
    #     result = sorted(inds)



class treehopper(hopper):
    def __init__(self, data, splits=float('inf'), metric=euclidean, inds=None,
                 partition = None,
                 max_partition_size=1000):

        t0 = time()
        self.times = []
        self.data = data
        self.numObs, self.numFeatures = data.shape

        self.r = float('inf')
        self.rs = [] #only upper-bounds the true Hausdorff, if using pre-partitions
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
        self.new = True

        if partition is not None:
            if callable(partition):
                '''use partitioner'''
                print('Pre-partitioning...')
                P = partition(data, max_partition_size, inds)
            else:
                P = partition
            for rows in P:
                h = hopper(data[rows,:], metric, inds=rows)
                h.hop() #hop once to set root
                # next = h.path_inds[-1]
                # self.path.append(next)
                # self.path_inds.append(inds[next])
                heappush(self.hheap, h)
            print('Pre-partitioning done, added {} points'.format(len(self.path)))

        self.init_time = time()-t0
        self.times = [self.init_time]

    def hop(self, n_hops=1, store_vcells=True):
        for _ in itertools.repeat(None, n_hops):
            t0 = time()
            print(len(self.path))
            if len(self.hheap) == 0: #start heaping
                print('heap starting')
                heappush(self.hheap, hopper(self.data,self.distfunc, range(self.numObs)))

            h = heappop(self.hheap)
            print('hopping with {} points'.format(h.numObs))
            print('radius {}'.format(h.r))


            if h.new: #first hop in this partition
                h.new = False # explored this one
                next = h.path_inds[-1]
                self.path.append(next)
                self.path_inds.append(self.inds[next])
                self.r = h.r
                self.rs.append(self.r)
                heappush(self.hheap, h)
                self.times.append(self.times[-1]+time()-t0)
                print('continuing')
                continue

            # if len(h.path) == 1: #first in a partition; add first two points
            #     next = h.path_inds[-1]
            #     self.path.append(next)
            #     self.path_inds.append(self.inds[next])
            #     self.r = h.r
            #     self.rs.append(self.r)
            #     h.hop()
            #     heappush(self.hheap, h)
            #     self.times.append(self.times[-1]+time()-t0)
            #     print('continuing')
            #     continue


            self.r = h.r
            self.rs.append(self.r)
            h.hop() #initialize stuff, or hop again


            next = h.path_inds[-1]

            self.path.append(next)
            self.path_inds.append(self.inds[next])

            if len(h.min_dists) > 0:
                if len(h.path) < self.splits:
                    heappush(self.hheap,h)

                else:
                    print('splitting')
                    #split into sub-hoppers
                    h.get_vdict()
                    for vcell in h.vdict.keys():
                        vcelldata = self.data[h.vdict[vcell],:]

                        #avail_idx = np.array(h.inds)[h.avail_inds].tolist()



                        #mindists = [0]+[h.min_dists[avail_idx.index(x)] for x in h.vdict[vcell][1:]]
                        #rad = h.rs[h.path_inds.index(vcell)]

                        inds = h.vdict[vcell]
                        #print(sorted(inds))

                        newhopper = hopper(vcelldata, metric=self.distfunc, inds=inds, root=0)

                        #newhopper.min_dists = mindists


                        newhopper.hop() #Initializes r, stops sampling the root
                        #print(newhopper.vdict.keys())

                        if len(newhopper.min_dists) > 0: #make sure not exhausted
                            heappush(self.hheap,newhopper)
            else:
                print('hopper exhausted!')

            self.times.append(self.times[-1]+time()-t0)

    def get_vdict(self):
        result = {}
        for h in self.hheap:
            if not h.new: #only update if it's been sampled
                result.update(h.get_vdict())
                print('adding {}'.format(len(h.get_vdict())))
                print('{} in path'.format(len(h.path)))
                print(len(result))

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
                'vdict':self.vdict, 'times': self.times, 'rs': self.rs}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def get_wts(self):
        if self.vcells is None:
            self.get_vcells()
        #See how many points each represents
        counter = Counter(self.vcells)
        self.wts = [counter[x] for x in hopper.path_inds]

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
            if 'wts' in hdata:
                self.wts = hdata['wts']
            self.times = hdata['times']
            self.rs = hdata['rs']
