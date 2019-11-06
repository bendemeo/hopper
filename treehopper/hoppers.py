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
class point:
    def __init__(self, coords, ind=None, alpha=float('inf'), r=float('inf')):
        self.ind = ind #index in full dataset
        self.coords = coords #representation
        self.r = r
        self.alpha = alpha

    def __lt__(self, other): #Make it randomized?
        # #prob = self.r / (self.r + other.r)
        # if self.r == float('inf'):
        #     return(True)
        # roll = np.random.uniform(high=self.r+other.r)
        #
        # prob = 1/(1+np.exp(-1 * self.alpha * (self.r-other.r)))
        # print(prob)
        # return (roll < self.r)

        return self.r > other.r #backwards, for min-heaping


@total_ordering
class vcell:
    def __init__(self, rep, pts, metric=euclidean, alpha=0):
        self.pts = pts
        self.rep = rep #rep point
        self.metric = metric
        self.alpha = alpha

        heapify(self.pts)
        if len(pts) > 0:
            self.r = pts[0].r # assumes points are sorted
        else:
            self.r = 0

    def dist(self, other):
        return(self.metric(self.rep.coords, other.rep.coords))

    def __lt__(self, other):

        prob = (1-self.alpha)*(self.pts[0]<other.pts[0])+ self.alpha*(len(self.pts)/(len(self.pts)+len(other.pts)))
        roll = np.random.uniform()
        print(prob)
        return(roll < prob)

        # pt1 = self.pts[0]
        # pt2 = other.pts[0]
        # prob = (len(self))/(len(self)+len(other))
        #
        #
        # return self.pts[0] < other.pts[0]


    def push(self,pt):
        heappush(self.pts, pt)
        self.r = self.pts[0].r

    def pop(self):
        result = heappop(self.pts)
        if len(self.pts) > 0:
            self.r = self.pts[0].r
        else:
            self.r = 0
        return(result)

    def order(self): #set radii and heapify
        data = np.array([pt.coords for pt in self.pts])
        dists = pairwise_distances(self.rep.coords.reshape((1,len(self.rep.coords))),
                                   data)[0,:]
        #print(dists[:10])
        for i,pt in enumerate(self.pts):
            pt.r = dists[i]

        heapify(self.pts)
        self.r = self.pts[0].r

    def __getitem__(self, key):
        return(self.pts[key])

@total_ordering
class hopper:
    def __init__(self, data, metric=euclidean, inds=None, start_r=float('inf'), alpha=0, root=None):

        t0 = time()
        self.times = [] # store runtimes after each hop
        self.r = start_r #current Haus distance
        self.rs = [] # keep track of haus after each time point

        self.alpha = alpha

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

        #self.min_dists = [] #dist to closest pt in traversal

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
        self.points_examined = []
        self.cells_examined = []

        start_pts = [point(data[i,:],inds[i], alpha=self.alpha) for i in range(data.shape[0])]

        start_cell = vcell(None, start_pts, metric=metric, alpha=self.alpha)

        self.cell_heap = [start_cell] # blank-slate cell with all points

    def hop(self, n_hops=1, store_vcells=True, alpha=0):
        for _ in itertools.repeat(None, n_hops):
            t0 = time()

            if len(self.path) == 0:
                cell = self.cell_heap[0] # all points

                print('beginning traversal! {} items to traverse'.format(self.numObs))
                #set starting point, initialize dist heap
                if self.root is None:
                    first = np.random.choice(list(range(len(cell.pts))))
                else:
                    first = self.root

                #first_heap = []


                cell.rep = cell.pts[first]
                self.path.append(cell.pts[first].ind)
                self.path_inds.append(self.inds[cell.pts[first].ind])

                del cell.pts[first] #rep not included in cell pts
                cell.order() # set rs and root

                self.r = cell[0].r
                self.r0 = cell[0].r

                print(cell[0].r)

                print(self.cell_heap[0].r)

            else:
                #print(self.min_dists[:5])
                total_checked = 0
                cells_checked = 0
                if len(self.cell_heap) < 1:
                    print('hopper exhausted!')
                    break

                print('heap size {}'.format(len(self.cell_heap)))

                #print(self.min_dists[0][2])
                top_cell = self.cell_heap[0]
                next_pt = top_cell.pop() #removes from cell


                #prune empty cell
                if len(top_cell.pts) == 0:
                    heappop(self.cell_heap)
                else:
                    top_cell.r = top_cell[0].r #update radius of this cell

                # add to sample
                self.path.append(next_pt.ind)
                self.path_inds.append(self.inds[next_pt.ind])

                new_cell = vcell(rep=next_pt, pts=[], metric=self.distfunc, alpha=self.alpha)

                to_delete = [] # for vcells that become empty
                cell_rad = float('inf')

                for i, cell in reversed(list(enumerate(self.cell_heap))):
                    #cell = self.cell_heap[i]

                    if cell.r <= (next_pt.r / 2.): #cell too small to change
                        continue

                    if new_cell.dist(cell) > (2*cell.r): # cell too far away
                        continue

                    cells_checked += 1

                    check_pts = []

                    r = cell[0].r
                    # print(r)
                    # print(cell.r)
                    # print(top_cell.r)
                    # print(next_pt.r)
                    while r > (next_pt.r/2.):
                        pt = cell.pop()
                        check_pts.append(pt)
                        if len(cell.pts) == 0:
                            break
                        r = cell[0].r

                    prev_dists = [x.r for x in check_pts]
                    check_coords = np.array([x.coords for x in check_pts])
                    #print(check_coords)
                    new_dists = pairwise_distances(next_pt.coords.reshape(1,-1), check_coords)[0,:]

                    total_checked += len(check_pts)
                    new_dists = np.array(new_dists)

                    for j in range(len(check_pts)):
                        pt = check_pts[j]
                        if new_dists[j] < prev_dists[j]:
                            #assign to new cell
                            pt.r = new_dists[j]
                            new_cell.push(pt)
                        else:
                            cell.push(pt)

                    if len(cell.pts) > 0:
                        #print([p.r for p in cell.pts])
                        cell.r = cell.pts[0].r #update to reflect lost points
                    else:
                        print('removing empty cell')
                        del self.cell_heap[i] #cell contains  no unsampled points; delete

                heapify(self.cell_heap) #to correct for lost values

                if len(new_cell.pts) > 0:
                    new_cell.r = new_cell.pts[0].r
                    heappush(self.cell_heap, new_cell)
                else:
                    print('new cell empty')

                self.cells_examined.append(cells_checked)
                self.points_examined.append(total_checked)
                print('sampled {}th point, checked {} points total, {} cells examined'.format(len(self.path), total_checked, cells_checked))

            #store Hausdorff and time information
            if len(self.cell_heap) < 1:
                self.r = 0
            else:
                #self.r = -1*self.min_dists[0][0]
                self.r = self.cell_heap[0].pts[0].r


            self.rs.append(self.r)
            self.times.append(self.times[-1]+time()-t0)

            #
            #         changed = list(itertools.compress(range(len(ischanged)),ischanged))
            #         unchanged = list(itertools.compress(range(len(ischanged)),1-np.array(ischanged)))
            #
            #         #print(len(unchanged))
            #         #print(len(cur_heap))
            #         for i in changed:
            #             new = new_dists[i]
            #             max_cell_rad = max([max_cell_rad, new])
            #             idx = check_list[i][2]
            #             noise = check_list[i][3]
            #
            #             if new == 0: #don't push rep point onto its own heap
            #                 print('not pushing')
            #                 continue
            #             heappush(new_heap, (-1*(new + noise), -1*new, idx, noise))
            #             self.vcells[idx] = self.inds[next_ind]
            #         for i in unchanged:
            #             heappush(cur_heap, check_list[i])
            #             max_cell_rad = max([max_cell_rad, prev_dists[i]])
            #
            #         if len(cur_heap) > 0:
            #             #print(cur_heap[:4])
            #             tup[0] = cur_heap[0][0] #update radius to reflect lost points
            #         else: # heap is irrelevant; delete it
            #             print('found irrelevant heap')
            #             del self.min_dists[j]
            #
            #     for i, c in reversed(list(enumerate(self.cell_heap))):
            #
            #
            #
            #
            #     # next_tup = heappop(self.min_dists[0][2])
            #     # next_ind = next_tup[2]
            #     #
            #     # if len(self.min_dists[0][2]) == 0:
            #     #     #cell now empty, remove from heap
            #     #     print('empty cell!')
            #     #     heappop(self.min_dists)
            #     # #next_ind = self.min_dists[0][2][0][2]
            #     # #print('ind: {}'.format(next_ind))
            #     #
            #     # cur_rad = -1*next_tup[1]
            #     #
            #     # #print(cur_rad)
            #     #
            #     # #next_ind = heappop(self.min_dists)[1]
            #     # next_pt = self.data[next_ind,:].reshape((1,self.numFeatures))
            #     #
            #     # self.path.append(next_ind)
            #     # self.path_inds.append(self.inds[next_ind])
            #     #
            #     # if store_vcells:
            #     #     self.vcells[next_ind]=self.inds[next_ind]
            #     #
            #     # new_heap = []  #heap for newly added point
            #     #
            #     # max_cell_rad = 0
            #     for j, tup in reversed(list(enumerate(self.min_dists))):
            #
            #         # print(tup[:2])
            #         cell_radius = -1*tup[0]
            #         #print(cell_radius)
            #         # print(cur_rad / 2)
            #         # print(cur_rad / 2.)
            #         if cell_radius < (cur_rad / 2): #can skip
            #              max_cell_rad = max([max_cell_rad, cell_radius])
            #              continue
            #
            #
            #         cur_ind = tup[1]
            #
            #         cur_pt = self.data[cur_ind,:].reshape((1,self.numFeatures))
            #
            #
            #         if self.distfunc(cur_pt, next_pt) > (2 * cell_radius):
            #             #too far away; ignore all points in this cell
            #             max_cell_rad = max([max_cell_rad, cell_radius])
            #             continue
            #         cells_checked += 1
            #         cur_heap = tup[2]
            #         #print(len(cur_heap))
            #
            #
            #         #find places where dists MAY be changed
            #         check_inds = [] # what indices to check
            #         check_list = [] # list of heap elements to check
            #         prev_dists = [] # prior distances
            #
            #         r = float('Inf')
            #         while r > cur_rad/2 and len(cur_heap) > 0:
            #             curtuple = heappop(cur_heap)
            #             check_inds.append(curtuple[2])
            #             check_list.append(curtuple)
            #             prev_dists.append(-1*curtuple[1])
            #             r = -1*curtuple[1]
            #
            #         #heappush(cur_heap, curtuple)
            #
            #         #print('checking {} points out of {}'.format(len(check_list), len(cur_heap)+len(check_list)))
            #
            #         #print(check_inds)
            #
            #         if len(check_inds) == 0: #nothing to do here; put stuff back
            #             continue
            #
            #         new_dists = pairwise_distances(np.array(next_pt), self.data[check_inds,:])[0,:]
            #
            #
            #         #print('{} of {}'.format(len(check_list), len(check_list)+len(cur_heap)))
            #         total_checked += len(check_list)
            #         #print(total_checked)
            #         new_dists = np.array(new_dists)
            #
            #         ischanged = (new_dists < prev_dists)
            #         changed = list(itertools.compress(range(len(ischanged)),ischanged))
            #         unchanged = list(itertools.compress(range(len(ischanged)),1-np.array(ischanged)))
            #
            #         #print(len(unchanged))
            #         #print(len(cur_heap))
            #         for i in changed:
            #             new = new_dists[i]
            #             max_cell_rad = max([max_cell_rad, new])
            #             idx = check_list[i][2]
            #             noise = check_list[i][3]
            #
            #             if new == 0: #don't push rep point onto its own heap
            #                 print('not pushing')
            #                 continue
            #             heappush(new_heap, (-1*(new + noise), -1*new, idx, noise))
            #             self.vcells[idx] = self.inds[next_ind]
            #         for i in unchanged:
            #             heappush(cur_heap, check_list[i])
            #             max_cell_rad = max([max_cell_rad, prev_dists[i]])
            #
            #         if len(cur_heap) > 0:
            #             #print(cur_heap[:4])
            #             tup[0] = cur_heap[0][0] #update radius to reflect lost points
            #         else: # heap is irrelevant; delete it
            #             print('found irrelevant heap')
            #             del self.min_dists[j]
            #
            #
            #     heapify(self.min_dists)
            #
            #     if len(new_heap) > 0:
            #         heappush(self.min_dists, [new_heap[0][0],next_ind, new_heap])
            #     else:
            #         print('new heap empty')
            #
            #     self.cells_examined.append(cells_checked)
            #     self.points_examined.append(total_checked)
            #     print('sampled {}th point, checked {} points total, {} cells examined'.format(len(self.path), total_checked, cells_checked))
            #     # print('checked {} points total'.format(total_checked))
            #     # print('checked {} cells of {}'.format(cells_checked,len(self.path)))
            #
            #
            #
            #     # #find places where dists MAY be changed
            #     # check_inds = [] # what indices to check
            #     # check_list = [] # list of heap elements to check
            #     # prev_dists = [] # prior distances
            #     #
            #     # r = float('inf')
            #     #
            #     # if len(self.min_dists) > 0:
            #     #     while r > self.r/2 and len(self.min_dists) > 0:
            #     #         curtuple = heappop(self.min_dists)
            #     #         check_inds.append(curtuple[1])
            #     #         check_list.append(curtuple)
            #     #         prev_dists.append(-1*curtuple[0])
            #     #         r = -1*curtuple[0]
            #     #
            #     #     heappush(self.min_dists, curtuple)
            #     #
            #     #     print('checking {} points'.format(len(check_list)))
            #     #
            #     #     #compute pairwise distances
            #     #     new_dists = pairwise_distances(np.array(next_pt), self.data[check_inds,:])[0,:]
            #     #     new_dists = np.array(new_dists)
            #     #
            #     #     #filter by changed vs. unchanged. Faster than looping
            #     #     ischanged = (new_dists < prev_dists)
            #     #     changed = list(itertools.compress(range(len(ischanged)),ischanged))
            #     #     unchanged = list(itertools.compress(range(len(ischanged)),1-np.array(ischanged)))
            #     #
            #     #     for i in changed:
            #     #         new = new_dists[i]
            #     #         idx = check_list[i][1]
            #     #         heappush(self.min_dists, (-1*new, idx))
            #     #         self.vcells[idx] = self.inds[next_ind]
            #     #     for i in unchanged:
            #     #         heappush(self.min_dists, check_list[i])
            #     # else:
            #     #     print('hopper exhausted!')
            #
            #
            # #store Hausdorff and time information
            # if len(self.min_dists) < 1:
            #     self.r = 0
            # else:
            #     #self.r = -1*self.min_dists[0][0]
            #     self.r = max_cell_rad
            # self.rs.append(self.r)
            # self.times.append(self.times[-1]+time()-t0)
            #

        return(self.path)


    def get_vcells(self):
        result = [None]*self.numObs
        for i, c in enumerate(list(self.cell_heap)):
            result[c.rep.ind] = c.rep
            for p in c.pts:
                result[p.ind] = c.rep

        self.vcells = result
        return(result)

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
                'times':self.times,'rs':self.rs, 'wts':self.wts,
                'points_examined':self.points_examined, 'cells_examined':self.cells_examined}
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
            if 'points_examined' in hdata:
                self.points_examined = hdata['points_examined']
            if 'cells_examined' in hdata:
                self.cells_examined = hdata['cells_examined']

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
        self.get_vcells()
        self.get_vdict()
        self.get_wts()

        data = {'path':self.path, 'vcells':self.vcells, 'path_inds':self.path_inds,
                'vdict':self.vdict, 'times': self.times, 'rs': self.rs,
                'wts':self.wts}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def get_wts(self):
        if self.vcells is None:
            self.get_vcells()
        #See how many points each represents
        counter = Counter(self.vcells)
        self.wts = [counter[x] for x in hopper.path_inds]
        return(self.wts)

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
