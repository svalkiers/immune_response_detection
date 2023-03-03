import numpy as np
import pandas as pd
import parmap
import time

from scipy.stats import hypergeom
from multiprocessing import Pool, cpu_count
from typing import Union

from .hashing import Cdr3Hasher, TCRDistEncoder
from .indexing import IvfIndex, FlatIndex
from .constants.sampling import match_vj_distribution
from .constants.datasets import sample_tcrs

def tcr_dict_to_df(neighbor_counts, cutoff=1, add_counts=False):
    neighbor_counts = {i:neighbor_counts[i] for i in neighbor_counts if neighbor_counts[i] > cutoff}
    vgenes = [i.split('_')[0] for i in list(neighbor_counts.keys())]
    cdr3aa = [i.split('_')[1] for i in list(neighbor_counts.keys())]
    if not add_counts:
        return pd.DataFrame({'v_call':vgenes, 'junction_aa':cdr3aa})
    else:
        counts = list(neighbor_counts.values())
        return pd.DataFrame({'v_call':vgenes, 'junction_aa':cdr3aa, 'neighbors':counts})

def assign_ids(query, lim):
    '''
    Match neighbor counts and indices.

    Parameters
    ----------
    query
        TCRs or CDR3 sequences for which the number of neighbors on the index
        was calculated.
    lim
        Faiss range search outcome that defines the number of retrieved neighbors
        for every queried vector.
    '''
    if isinstance(query, list):
        return dict([(j, int(lim[i+1]-lim[i])) for i,j in enumerate(query)])
    elif isinstance(query, pd.DataFrame):
        indices = (query['v_call'] + "_" + query['junction_aa']).to_list()
        return dict([(j, int(lim[i+1]-lim[i])) for i,j in enumerate(indices)])

def index_neighbors(query, r, index):
    '''
    Compute the number of neighbors on the index that are found within 
    a radius r of each query TCR. Returns a dict of query sequences
    and their respective neighbor count.

    Parameters
    ----------
    query
        TCRs or CDR3 sequences for which the number of neighbors on the index
        should be calculated.
    r
        Radius parameter that defines the edge of the TCR neighborhood.
    index
        Index on which the search should be performed. The index needs to be
        trained and should contain vectors to query against.
    '''
    X = index.hasher.fit_transform(query)
    lim, D, I = index.idx.range_search(X.astype(np.float32), thresh=r)
    return assign_ids(query=index.hasher.tcrs, lim=lim)

class NeighborEnrichment():
    def __init__(
        self,
        repertoire:Union[pd.DataFrame, list],
        hasher:Union[Cdr3Hasher, TCRDistEncoder],
        radius:Union[int,float]=12.5,
        # custom_background=None, -> add this functionality back in later
        # background_size=1e6,
        # ncpus:int=1 -> this does not seem to have any influence on the computational performance
        ):
        '''
        Class for calculating and statistically evaluating the neighbor count of 
        TCRs or CDR3 sequences.

        Parameters
        ----------
        repertoire: Union[pd.DataFrame, list]
            Repertoire of interest. This can be a list of CDR3 sequences
            as well as a pd.DataFrame containing at least V and CDR3
            information. Column names should satisfy AIRR data conventions.
        hasher: Union[Cdr3Hasher, TCRDistEncoder]
            Transformer function used to embed sequences into vector space.
        radius: Union[int,float]
            Radius r that determines the edge of the TCR neighborhood.
        
        '''
        self.repertoire = repertoire
        # self.background = custom_background -> add this functionality back in later
        self.hasher = hasher
        self.rsize = len(repertoire)
        self.r = radius

        self.hasher.fit()

        if ncpus == -1: # if set to -1, use all CPUs
            self.ncpus = cpu_count()
        else:
            self.ncpus = ncpus

    def compute_neighbors(self, exhaustive:bool=True, k=None, n=None):
        '''
        Find the number of neighbors within a repertoire of interest.

        Parameters
        ----------
        exhaustive: bool
            When True, performs an exhaustive search (using a faiss.IndexFlatL2) 
            to find all neighbors in the repertoire. Uses the faiss.IndexIVFFlat
            for a non-exhaustive alternative.
        k
            Number of centroids for fitting faiss.IndexIVFFlat.
        n
            Number of cells to probe when using the faiss.IndexIVFFlat.
        '''
        if exhaustive:
            self.fg_index = FlatIndex(hasher=self.hasher)
        else:
            if k is None:
                k = np.round(self.rsize/1000,0)
            if n is None:
                n = np.round(k/50,0)
            self.fg_index = IvfIndex(hasher=self.hasher, n_centroids=int(k), n_probe=int(n))
        self.fg_index.add(self.repertoire)
        self.nbr_counts = index_neighbors(query=self.repertoire, index=self.fg_index, r=self.r)

    def compute_pvalues(self, prefilter=True, depth=1000001, fdr=1):
        '''
        Assign p-values to neighbor counts by contrasting against a background distribution.
        Uses a prefilter step (if True) to limit the number of vectors to query on the index.

        Parameters
        ----------
        prefilter: bool
            If True, uses a prefiltering step that eliminates TCRs whose foreground neighbor count
            is less than or equal to that of a size-matched background repertoire.
        depth
            Size of the background repertoire.
        fdr
            False discovery rate threshold.
        '''
        # Neighbor counts in foreground should be determined
        assert self.nbr_counts is not None, 'Please compute foreground neighbors first.'

        print('retrieving background neighbors')
        k = int(depth/1000)
        bg_index = IvfIndex(hasher=self.hasher, n_centroids=k, n_probe=5)
        if prefilter:
            print('prefiltering')
            bg = match_vj_distribution(n=self.rsize, foreground=self.repertoire)
            bg_index.add(bg)
            del bg
            seq_with_nbrs = tcr_dict_to_df(self.nbr_counts, add_counts=True)
            nbrs_in_background = index_neighbors(query=seq_with_nbrs, index=bg_index, r=self.r)
            nbrs_in_background = tcr_dict_to_df(nbrs_in_background, add_counts=True)

            merged = seq_with_nbrs.merge(nbrs_in_background, on=['v_call', 'junction_aa'])
            filtered = merged[merged['neighbors_x']>merged['neighbors_y']]
            filtered = filtered.drop(columns=['neighbors_y']).rename(columns={'neighbors_x':'neighbors'})
            bg_index.n_probe = 10
            print('finished prefiltering')

        # Bigger background sample
        bg = match_vj_distribution(n=depth, foreground=self.repertoire)
        bg_index.add(bg)
        del bg
        # Find neighbors in background
        nbrs_in_background = index_neighbors(query=filtered, index=bg_index, r=self.r)
        nbrs_in_background = tcr_dict_to_df(nbrs_in_background, add_counts=True)

        col_remap = {'neighbors_x':'foreground_neighbors', 'neighbors_y':'background_neighbors'}
        merged = filtered.merge(nbrs_in_background, on=['v_call', 'junction_aa'])
        merged = merged.rename(columns=col_remap)

        # Hypergeometric testing
        M = len(bg_index.ids) + self.rsize
        N = self.rsize
        print('testing')
        merged['pval'] = merged.apply(
            lambda x: hypergeom.sf(x['foreground_neighbors']-1, M, x['foreground_neighbors'] + x['background_neighbors'], N),
            axis=1
            )
        merged = merged.sort_values(by='pval')
        return merged[merged.pval<=fdr]


    # def adaptive_sampling(self, fdr_threshold:float=1e-08, depth=1e6, regime=None):
    #     '''
    #     Iteratively sample deeper into the background and eliminate
    #     TCRs that do not satisfy the significance cut-off.
    #     '''
    #     # Find neighbors in foreground repertoire
    #     print('Computing neighbours in query repertoire...')
    #     nf = self.find_foreground_neighbors()
    #     if regime is None:
    #         sampling_regime = [1e5, 2e5, 5e5, 1e6, depth]
    #     else:
    #         sampling_regime = regime
    #     for n in sampling_regime:
    #         print(f'Sampling at depth: {n}')
    #         if n > depth:
    #             print(f'Sampling depth {depth} reached')
    #             break
    #         else:
    #             self._background(int(n))
    #         sign = self.neighbor_significance(nf)
    #         filtered = sign[(sign.bg_n == 0) | (sign.pvalue < 0.0005)]
    #         nf = {i:j for i,j in zip(filtered.sequence,filtered.fg_n)}
    #         n_elim = len(sign) - len(filtered)
    #         print(f'Eliminated {n_elim} TCRs,\n{len(nf)} remaining.')
    #     return sign