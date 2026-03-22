import scanpy as sc
import numpy as np
from collections import Counter




def compress(h, data):
    path = h.path_inds
    result = data[path,:]
    return(result)

def expand(h, fulldata):
    #use a hopper's data to nearest-neighbor classify full data
    #way to retain attributes of small data e.g. louvain clusters?
    #all attributes stored in the full data are kept!

    if h.vdict is None:
        h.get_vdict()

    inds = []

    for c in h.vdict.keys():
        inds += h.vdict[c]

    #print(inds)
    return fulldata[inds,:]


##WORK IN PROGRESS

# def expand_attr(vals, h):
#     #take any value e.g. louvain clustering and expand it
#
#
#     if h.vdict is None:
#         h.get_vdict()
#
#     expanded = []
#     for k in h.vdict.keys()
#     expanded = [None]*sum([len(h.vdict[k]) for k in h.vdict.keys()])
#
#     for c in h.vdict.keys():
#         inds.append(h.vdict[c])



def expand_clusterings(smalldata, fulldata, vdict=None, vcells=None, cluster_name='louvain', new_cluster_name=None, clusters = None, vc_name='vcell'):

    if new_cluster_name is None:
        new_cluster_name = cluster_name

    newclusts = np.array([None]*fulldata.obs.shape[0])
    if vdict is None: #build reverse lookup
        vdict = {}

        for i,c in enumerate(vcells):
            if c not in vdict:
                vdict[c] = [i]
            else:
                vdict[c].append(i)

    for i, cell in enumerate(list(smalldata.obs[vc_name])):
        print(i)
        inds = vdict[cell]
        # print(inds)
        # print(list(smalldata.obs[cluster_name].iloc[i]))
        # print(smalldata.obs[cluster_name])
        newclusts[inds]=len(inds)*[smalldata.obs[cluster_name].iloc[i]]

    fulldata.obs[new_cluster_name] = newclusts
    return(fulldata)

    # if clusters is none:
    #     clusters = np.unique(smalldata.obs[cluster_name])
    #
    #
    # for c in clusters:
    #     small = subset(smalldata, cluster_name, c)
    #     big = expand()


def subset(adata, obs_key, obs_values):
    obs_vals = list(adata.obs[obs_key])

    idx = np.where([x in obs_values for x in obs_vals])[0]

    return(adata[idx,:])


def viz(adata, rep = '',louvain=True, rerun=False, **kwargs):
    '''visualize an AnnData object using UMAP and Louvain'''

    if rep is None:
        rep='X'
    if rep == 'X':
        suffix=''
    else:
        suffix=rep

    print('computing neighbor graph...')
    if rerun or 'neighbors'.format(rep) not in adata.uns:
        sc.pp.neighbors(adata, use_rep=rep)

    print('running UMAP...')
    if rerun or 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)

    print('Louvain clustering...')
    if rerun or 'louvain' not in adata.obs:
        sc.tl.louvain(adata)

    sc.pl.umap(adata, **kwargs)



# def compress(adata, hopper, vc_name = 'vcell', wt_name='wt'):
#     '''given voronoi cells, make weighted dataset'''
#
#     vcells = hopper.vcells
#     path = hopper.path_inds
#
#     #add voronoi info to original data
#     adata.obs[vc_name] = vcells
#
#     #compute counts in each cell
#     counter = Counter(vcells)
#     #print(counter)
#     #print(hopper.path_inds)
#     wts = [counter[x] for x in hopper.path_inds]
#
#     result = adata[path,:]
#     result.obs[wt_name] = wts
#
#     return(result)

# def expand(smalldata, fulldata, vdict=None, vc_name = 'vcell'):
#     '''given a small dataset with compression info, expand to full points,
#     keeping all observation data (e.g. clusters)'''
#
#     smallcells = list(smalldata.obs[vc_name])
#     fullcells = list(fulldata.obs[vc_name])
#     print(smallcells)
#     print(fullcells)
#
#     idx = np.where([x in smallcells for x in fullcells])[0]
#
#     result = fulldata[idx,:]
#     # for o in smalldata.obs.columns:
#     #     result.obs[o] = [None]*len(idx)
#     #
#     #
#     # for i,x in enumerate(smallcells):
#     #     print(i)
#     #     inds = np.where([y==x for y in fullcells])
#     #     for o in smalldata.obs.columns:
#     #         print(o)
#     #         result.obs[o].iloc[inds] = list(smalldata.obs[o].iloc[[i]])*len(inds)
#
#
#     return(result)
