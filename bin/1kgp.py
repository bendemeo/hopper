import allel
from anndata import AnnData
from fbpca import pca
from geosketch import gs
import numpy as np
import scanpy as sc
import sys
from treehopper import hopper

def samples_to_populations(pop_fname, samples):
    sample2pop = {}
    with open(pop_fname) as f:
        f.readline() # Consume header.
        for line in f:
            fields = line.rstrip().split()
            sample2pop[fields[0]] = fields[1]

    return [ sample2pop[sample] if sample in sample2pop else 'NA'
             for sample in samples ]

if __name__ == '__main__':

    vcfname = ('data/1kgp/ALL.wgs.nhgri_coriell_affy_6.20140825.'
               'genotypes_has_ped.vcf.gz')
    pop_fname = 'data/1kgp/affy_samples.20141118.panel'

    v = allel.read_vcf(vcfname, alt_number=1)
    print('Done reading VCF into memory.')

    genotype = v['calldata/GT'].astype(float)
    genotype[genotype < 0] = np.nan
    genotype = genotype.sum(2)
    genotype = genotype[np.logical_not(np.isnan(genotype.sum(1)))]
    genotype = genotype.T

    print('Found {} samples and {} SNPs'.format(*genotype.shape))

    U, s, Vt = pca(genotype, k=100)

    X_pca = U * s

    populations = np.array(samples_to_populations(
        pop_fname, v['samples']
    ))

    adata = AnnData(
        genotype,
        { 'id': v['samples'], 'population': populations },
    )
    adata.obsm['X_pca'] = X_pca

    sc.tl.tsne(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.pca(adata, color='population', save='_full.png')
    sc.pl.tsne(adata, color='population', save='_full.png')
    sc.pl.umap(adata, color='population', save='_full.png')

    hop_obj = hopper(X_pca, verbose=False)

    sketch_sizes = [ 50, 100, 200, 300 ]
    for i in range(len(sketch_sizes)):
        if i == 0:
            n_hops = sketch_sizes[i]
        else:
            n_hops = sketch_sizes[i] - sketch_sizes[i - 1]
        hop_obj.hop(n_hops=n_hops)

        adata = AnnData(X_pca[hop_obj.path_inds],
                        { 'population': populations[hop_obj.path_inds] })
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata)
        sc.pl.umap(adata, color='population',
                   save='_hopper{}.png'.format(sketch_sizes[i]))

        gs_idx = gs(X_pca, sketch_sizes[i])
        adata = AnnData(X_pca[gs_idx],
                        { 'population': populations[gs_idx] })
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata)
        sc.pl.umap(adata, color='population',
                   save='_geosketch{}.png'.format(sketch_sizes[i]))
