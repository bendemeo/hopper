"""Visualization and expansion utilities for use with scanpy/AnnData.

These functions require the optional ``scanpy`` dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def compress(h, data):
    """Subset ``data`` to the sketch points."""
    return data[h.path_inds, :]


def expand(h, fulldata):
    """Expand the sketch back to the full dataset using Voronoi cells."""
    if h.vdict is None:
        h.get_vdict()

    inds = []
    for c in h.vdict.keys():
        inds += h.vdict[c]

    return fulldata[sorted(inds), :]


def expand_clusterings(
    smalldata: AnnData,
    fulldata: AnnData,
    vdict=None,
    vcells=None,
    cluster_name="louvain",
    new_cluster_name=None,
    vc_name="vcell",
):
    """Propagate cluster labels from a sketch to the full dataset via Voronoi cells.

    Parameters
    ----------
    smalldata : AnnData
        The sketched dataset with cluster labels in ``.obs``.
    fulldata : AnnData
        The full dataset to propagate labels to.
    vdict : dict, optional
        Pre-computed Voronoi dictionary.
    vcells : list, optional
        Per-point Voronoi cell assignments (used to build ``vdict`` if not given).
    cluster_name : str
        Column name in ``smalldata.obs`` containing cluster labels.
    new_cluster_name : str, optional
        Column name for the propagated labels in ``fulldata.obs``.
    vc_name : str
        Column name for Voronoi cell IDs in ``smalldata.obs``.

    Returns
    -------
    AnnData
        ``fulldata`` with the new cluster column added.
    """
    if new_cluster_name is None:
        new_cluster_name = cluster_name

    newclusts = np.array([None] * fulldata.obs.shape[0])

    if vdict is None:
        vdict = {}
        for i, c in enumerate(vcells):
            if c not in vdict:
                vdict[c] = [i]
            else:
                vdict[c].append(i)

    for i, cell in enumerate(list(smalldata.obs[vc_name])):
        inds = vdict[cell]
        newclusts[inds] = len(inds) * [smalldata.obs[cluster_name].iloc[i]]

    fulldata.obs[new_cluster_name] = newclusts
    return fulldata


def subset(adata: AnnData, obs_key: str, obs_values) -> AnnData:
    """Subset an AnnData object by observation metadata values."""
    obs_vals = list(adata.obs[obs_key])
    idx = np.where([x in obs_values for x in obs_vals])[0]
    return adata[idx, :]


def viz(adata: AnnData, rep: str = "", louvain: bool = True, rerun: bool = False, **kwargs) -> None:
    """Visualize an AnnData object using UMAP and Louvain clustering.

    Parameters
    ----------
    adata : AnnData
        The dataset to visualize.
    rep : str
        Representation to use for neighbor computation.
    louvain : bool
        Whether to run Louvain clustering.
    rerun : bool
        Force recomputation even if results exist.
    **kwargs
        Passed to ``scanpy.pl.umap``.
    """
    import scanpy as sc

    if rep is None:
        rep = "X"
    if rep == "X":
        suffix = ""
    else:
        suffix = rep

    if rerun or "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=rep)

    if rerun or "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    if louvain and (rerun or "louvain" not in adata.obs):
        sc.tl.louvain(adata)

    sc.pl.umap(adata, **kwargs)
