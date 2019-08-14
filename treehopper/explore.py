import scanpy as sc
import numpy as np
from collections import Counter


def compress(adata, hopper, vc_name = 'vcell', wt_name='wt'):
    '''given voronoi cells, make weighted dataset'''

    vcells = hopper.vcells
    path = hopper.path

    #add voronoi info to original data
    adata.obs[vc_name] = vcells

    #compute counts in each cell
    counter = Counter(vcells)
    #print(counter)
    #print(hopper.path_inds)
    wts = [counter[x] for x in hopper.path_inds]

    result = adata[path,:]
    result.obs[wt_name] = wts

    return(result)


def expand(smalldata, fulldata, vc_name = 'vcell'):
    '''given a small dataset with compression info, expand to full points,
    keeping all observation data (e.g. clusters)'''

    smallcells = list(smalldata.obs[vc_name])
    fullcells = list(fulldata.obs[vc_name])
    print(smallcells)
    print(fullcells)

    idx = np.where([x in smallcells for x in fullcells])[0]

    result = fulldata[idx,:]
    for o in smalldata.obs.columns:
        result.obs[o] = [None]*len(idx)


    for i,x in enumerate(smallcells):
        print(i)
        inds = np.where([y==x for y in fullcells])
        for o in smalldata.obs.columns:
            result.obs[o][inds,:] = list(smalldata.obs[o][i])*len(inds)

    return(result)

def subset(adata, obs_key, obs_values):
    obs_vals = list(adata.obs[obs_key])

    idx = np.where([x in obs_values for x in obs_vals])[0]

    return(adata[idx,:])


def viz(adata, rep = 'X_ica',louvain=True, **kwargs):
    print('computing neighbor graph...')
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=rep)

    print('running UMAP...')
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)

    print('Louvain clustering...')
    if 'louvain' not in adata.obs:
        sc.tl.louvain(adata)

    sc.pl.umap(adata, **kwargs)
