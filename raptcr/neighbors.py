import numpy as np
import pandas as pd
import parmap
import time

from scipy.stats import hypergeom
from multiprocessing import Pool, cpu_count
from typing import Union

from .hashing import Cdr3Hasher, TCRDistEncoder
from .indexing import IvfIndex
from .analysis import TcrCollection
from .constants.datasets import sample_cdr3s

def nneighbors(seq, index, r:Union[float,int], exclude_self:bool=True) -> tuple:
    '''
    Retrieve the number of neighbors within a radius r for a given seq.
    '''
    nneigh = len(index.radius_search(seq, r=r, exclude_self=exclude_self))
    return (seq, nneigh)

def find_neighbors_within_radius(query, dataset, hasher:Cdr3Hasher, r:Union[float,int], k:int=150, p:int=3, ncpus:int=1):
    '''
    Retrieve number of neighbors within radius r in dataset for all sequences in query. 

    Parameters
    ----------
    dataset
        Vectors to store on index.
    k
        Number of centroids to compute during index construction.
    p
        Number of cells to probe during index search.
    '''
    
    # Set up the index
    t0 = time.time()
    index = IvfIndex(hasher=hasher, n_centroids=k, n_probe=p)
    index.add(dataset)
    print(f'Time elapsed building index: {np.round(time.time()-t0, 2)} s.')

    # if ncpus > 1:
    #     '''
    #     !!!
    #     Seems to work slower than running on single core
    #     Check whether faiss uses parallel processing under the hood for index searching

    #     From the faiss documentation:
    #     Faiss itself is internally threaded in a couple of different ways. 
    #     For CPU Faiss, the three basic operations on indexes (training, adding, searching) are internally multithreaded

    #     --> Remove this functionality as it is redundant and results in more overhead
    #     '''
    #     with Pool(ncpus) as pool:counts = [nneighbors(seq=seq, index=index, r=0.1, exclude_self=True) for seq in query]
    #         counts = parmap.map(
    #             nneighbors,
    #             query,
    #             index=index,
    #             r=.1,
    #             exclude_self=True
    #             )
    # else:

    if isinstance(query, list):
        t0 = time.time()
        counts = [nneighbors(seq=seq, index=index, r=r, exclude_self=True) for seq in query]
        print(f'Time elapsed finding neighbors: {time.time()-t0} s.')
        return dict(counts)
    elif isinstance(query, pd.DataFrame):
        counts = index.array_search(query=query, r=r, exclude_self=True)
        return counts.source.value_counts().to_dict()
        # counts = [nneighbors(seq=pd.DataFrame(seq[1]).T, index=index, r=r, exclude_self=True) for seq in query.iterrows()]

class NeighborEnrichment():
    def __init__(
        self,
        repertoire:Union[TcrCollection, pd.DataFrame, list],
        hasher:Union[Cdr3Hasher, TCRDistEncoder],
        radius:Union[int,float]=12.5,
        n_centroids:int=1000,
        n_probe:int=50,
        custom_background=None,
        ncpus:int=1
        ):

        self.repertoire = repertoire
        self.background = custom_background
        self.hasher = hasher
        self.rsize = len(repertoire)
        self.n_centroids = n_centroids
        self.n_probe = n_probe
        self.r = radius

        if ncpus == -1: # if set to -1, use all CPUs
            self.ncpus = cpu_count()
        else:
            self.ncpus = ncpus

    def _background(self, n):
        '''
        Generate a background dataset by randomly sampling n CDR3 sequences.
        '''
        if self.background is None:
            self.background = sample_cdr3s(n)

    def find_foreground_neighbors(self, eliminate_singlets:bool=True):
        fg = find_neighbors_within_radius(
            query=self.repertoire, 
            dataset=self.repertoire, 
            hasher=self.hasher, 
            r=self.r,
            k=self.n_centroids,
            p=self.n_probe
            )
        if eliminate_singlets:
            return {i:fg[i] for i in fg if fg[i] > 0}
        else:
            return fg

    def neighbor_significance(self, foreground_counts:dict):
        '''
        Hypergeometric test to determine whether a TCR has more neighbors
        than would be expected by sampling from a random background sample.
        '''
        # Sequences
        q = list(foreground_counts.keys())
        # Find n neighbors of query in background
        bg = find_neighbors_within_radius(query=q, hasher=self.hasher, r=self.r, dataset=self.background)
        # Neighbor counts
        n_b = list(bg.values()) # in background
        n_f = list(foreground_counts.values()) # in foreground
        # Hypergeometric testing
        n = self.rsize
        M = len(self.background) + n
        pvals = [hypergeom.sf(i, M, n, i+j) for i,j in zip(n_f,n_b)]
        return pd.DataFrame({'sequence':q, 'fg_n':n_f, 'bg_n':n_b, 'pvalue':pvals})

    def adaptive_sampling(self, depth=1e6, regime=None):
        '''
        Iteratively sample deeper into the background and eliminate
        TCRs that do not satisfy the significance cut-off.
        '''
        # Find neighbors in foreground repertoire
        print('Computing neighbours in query repertoire...')
        nf = self.find_foreground_neighbors()
        if regime is None:
            sampling_regime = [1e5, 2e5, 5e5, 1e6, depth]
        else:
            sampling_regime = regime
        for n in sampling_regime:
            print(f'Sampling at depth: {n}')
            if n > depth:
                break
            else:
                self._background(int(n))
            sign = self.neighbor_significance(nf)
            filtered = sign[(sign.bg_n == 0) | (sign.pvalue < 0.05)]
            nf = {i:j for i,j in zip(filtered.sequence,filtered.fg_n)}
        return sign