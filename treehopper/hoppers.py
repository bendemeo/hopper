import numpy as np
import itertools
from functools import total_ordering
from scipy.spatial.distance import euclidean
from heapq import heappush, heappop, heapify, heapreplace
from sklearn.metrics import pairwise_distances
import pickle
from time import time



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

        t0 = time()
        self.times = [] # store runtimes after each hop

        self.r = start_r #current Haus distance
        self.rs = [] # keep track of haus after each time point
        self.numObs, self.numFeatures = data.shape

        if inds is None:
            inds = range(self.numObs)

        self.inds = inds
        self.data = data
        self.path = []
        self.path_inds = []

        self.min_dists = []

        avail_inds = list(range(self.numObs))
        self.avail_inds = avail_inds

        self.distfunc = metric
        self.vcells=None
        self.vdict=None
        self.root = root
        self.init_time = time()-t0
        self.times.append(self.init_time)


    def hop(self, n_hops=1, store_vcells=True):
        '''generate exact far traversal'''

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

                for ind in range(self.numObs):
                    if ind != first:
                        heappush(self.min_dists, (-1*start_dists[ind], ind))


                if store_vcells:
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

                #
                # for i,tuple in enumerate(check_list):
                #     new = new_dists[i]
                #     prev = prev_dists[i]
                #     idx = tuple[1]
                #     if new < prev:
                #         heappush(self.min_dists, (-1*new, idx))
                #         self.vcells[idx] = self.inds[next_ind]
                #
                #     else: #no change; put it back
                #         heappush(self.min_dists, tuple)

                # #places where dists MAY be changed
                # check = np.array(self.min_dists) > float(self.r)/2
                # check_pos = list(itertools.compress(range(len(self.avail_inds)),check))
                # check_inds = np.array(self.avail_inds)[check_pos]
                #
                # print('{} indices to check'.format(len(check_inds)))
                #
                # #compute distances
                # dists = pairwise_distances(np.array(next_pt), self.data[check_inds,:])[0,:]
                # dists = np.array(dists)
                #
                # #find places where distance WILL be changed
                # prev_dists = np.array(self.min_dists)[check_pos]
                # change = (dists < prev_dists)
                #
                # change_pos = list(itertools.compress(range(len(check_pos)),change))
                # change_inds = np.array(check_pos)[change_pos]
                #
                # #update minimum distances and vcells, if applicable
                # for i, idx in enumerate(change_pos):
                #     #print(self.min_dists[idx])
                #     self.min_dists[change_inds[i]] = dists[idx]
                #     self.closest[change_inds[i]] = len(self.path) - 1
                #     # self.r = max(self.r, )
                #     # self.rs[len(self.path)-1] = max(self.rs[len(self.path)-1],
                #     #                                 self.min_dists[idx])
                #     if store_vcells:
                #         self.vcells[check_inds[change_pos[i]]] = self.inds[next_ind]
                #
                #
                # #This line may be a bit slow -- consider sorting min_dists?
                # self.max_pos = self.min_dists.index(max(self.min_dists))
                # self.r = self.min_dists[self.max_pos]

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
        self.vdict = result
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

                        avail_idx = np.array(h.inds)[h.avail_inds].tolist()



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
