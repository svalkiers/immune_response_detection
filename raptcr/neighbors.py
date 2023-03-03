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
from .sampling import match_vj_distribution
from .constants.datasets import sample_cdr3s, sample_tcrs

def tcr_dict_to_df(neighbor_counts, cutoff=1, add_counts=False):
    neighbor_counts = {i:neighbor_counts[i] for i in neighbor_counts if neighbor_counts[i] > cutoff}
    vgenes = [i.split('_')[0] for i in list(neighbor_counts.keys())]
    cdr3aa = [i.split('_')[1] for i in list(neighbor_counts.keys())]
    if not add_counts:
        return pd.DataFrame({'v_call':vgenes, 'junction_aa':cdr3aa})
    else:
        counts = list(neighbor_counts.values())
        return pd.DataFrame({'v_call':vgenes, 'junction_aa':cdr3aa, 'neighbors':counts})

def nneighbors(seq, index, r:Union[float,int], exclude_self:bool=True) -> tuple:
    '''
    Retrieve the number of neighbors within a radius r for a given seq.
    '''
    nneigh = len(index.radius_search(seq, r=r, exclude_self=exclude_self))
    return (seq, nneigh)

def find_neighbors_within_radius(query, dataset, hasher:Cdr3Hasher, r:Union[float,int], index=None, k:int=150, p:int=3, ncpus:int=1):
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
    
    # Set up the index if none provided
    if index is None:
        index = IvfIndex(hasher=hasher, n_centroids=k, n_probe=p)
        index.add(dataset)
    else:
        pass

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
        X = hasher.fit_transform(query)
        lims, D, I = index.idx.range_search(X, thresh=r)
        return dict([(j, int(lims[i+1]-lims[i])) for i,j in enumerate(query)])
        # t0 = time.time()
        # counts = [nneighbors(seq=seq, index=index, r=r, exclude_self=True) for seq in query]
        # print(f'Time elapsed finding neighbors: {time.time()-t0} s.')
        # return dict(counts)
    elif isinstance(query, pd.DataFrame):
        counts = index.array_search(query=query, r=r, exclude_self=True)
        return counts.source.value_counts().to_dict()
        # counts = [nneighbors(seq=pd.DataFrame(seq[1]).T, index=index, r=r, exclude_self=True) for seq in query.iterrows()]

def assign_ids(query, lim):
    if isinstance(query, list):
        return dict([(j, int(lim[i+1]-lim[i])) for i,j in enumerate(query)])
    elif isinstance(query, pd.DataFrame):
        indices = (query['v_call'] + "_" + query['junction_aa']).to_list()
        return dict([(j, int(lim[i+1]-lim[i])) for i,j in enumerate(indices)])

def index_neighbors(query, r, index):
    X = index.hasher.fit_transform(query)
    lim, D, I = index.idx.range_search(X.astype(np.float32), thresh=r)
    return assign_ids(query=query, lim=lim)
    

class NeighborEnrichment():
    def __init__(
        self,
        repertoire:Union[TcrCollection, pd.DataFrame, list],
        hasher:Union[Cdr3Hasher, TCRDistEncoder],
        radius:Union[int,float]=12.5,
        # n_centroids:int=1000,
        # n_probe:int=50,
        custom_background=None,
        background_size=1e6,
        ncpus:int=1
        ):

        self.repertoire = repertoire
        self.background = custom_background
        self.background_size = int(background_size)
        self.hasher = hasher
        self.rsize = len(repertoire)
        self.r = radius

        self.hasher.fit()
        # self.hashes = self.hasher.transform(self.repertoire)

        if ncpus == -1: # if set to -1, use all CPUs
            self.ncpus = cpu_count()
        else:
            self.ncpus = ncpus

        # if self.background is None:
        #     self.background()
        # self.index = IvfIndex(hasher=self.hasher, n_centroids=)

    def _background(self):
        '''
        Generate a background dataset by randomly sampling n CDR3 sequences.
        '''
        if isinstance(self.repertoire, list):
            return sample_cdr3s(self.background_size)
        elif isinstance(self.repertoire, pd.DataFrame):
            return sample_tcrs(self.background_size)

    def neighbors_in_foreground(self):
        k = int(self.rsize/200)
        n = int(k/20)
        index = IvfIndex(self.hasher, n_centroids=k, n_probe=n)
        index.add(self.repertoire)
        return index_neighbors(query=self.repertoire, r=self.r, index=index)

    def neighbors_in_background(self, sequences, r:Union[int,float], index=None, k:int=None, n:int=None):
        if k is None:
            k = int(self.background_size/200)
        if n is None:
            n = int(k/50)
        bg = sample_tcrs(self.background_size)
        if index is None:
            index = IvfIndex(self.hasher, n_centroids=k, n_probe=n)
        else:
            pass
        index.add(bg)
        del bg
        return index_neighbors(query=sequences, r=r, index=index)

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

    def compute_neighbors(self, exhaustive=True, k=None, n=None):

        if exhaustive:
            fg_index = FlatIndex(hasher=self.hasher)
        else:
            if k is None:
                k = np.round(self.rsize/1000,0)
            if n is None:
                n = np.round(k/50,0)
            fg_index = IvfIndex(hasher=self.hasher, n_centroids=int(k), n_probe=int(n))

        fg_index.add(self.repertoire)
        self.nbr_counts = index_neighbors(query=self.repertoire, index=fg_index, r=self.r)

    def calculate_pvalues(self, prefilter=True, depth=1000001, fdr=1e-5):

        assert self.nbr_counts is not None, 'Please compute foreground neighbors first.'

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
        bg = match_vj_distribution(n=depth, foreground=fg)
        bg_index.add(bg)
        del bg
        # Find neighbors in background
        nbrs_in_background = index_neighbors(query=filtered, index=bg_index, r=self.r)
        nbrs_in_background = tcr_dict_to_df(nbrs_in_background, add_counts=True)
        # Hypergeometric testing
        merged = filtered.merge(nbrs_in_background, on=['v_call', 'junction_aa'])
        M = len(bg_index.ids) + self.rsize
        N = self.rsize
        print('testing')
        merged['pval'] = merged.apply(
            lambda x: hypergeom.sf(x['neighbors_x']-1, M, x['neighbors_x'] + x['neighbors_y'], N),
            axis=1
            )
        col_remap = {'neighbors_x':'foreground_neighbors', 'neighbors_y':'background_neighbors'}
        merged = merged.rename(columns=col_remap)
        return merged.sort_values(by='pval')

    def neighbor_significance(self, foreground_counts:dict=None, background_counts:dict=None):
        '''
        Hypergeometric test to determine whether a TCR has more neighbors
        than would be expected by sampling from a random background sample.
        '''
        # # Sequences
        # q = list(foreground_counts.keys())
        # if len(q[0].split('_')) == 2:
        #     q = pd.DataFrame(foreground_counts.items(), columns=['tcr','nbrs'])
        #     q['v_call'] = q.tcr.apply(lambda x: x.split('_')[0])
        #     q['junction_aa'] = q.tcr.apply(lambda x: x.split('_')[1])
        # bgrep = self._background()
        # # Find n neighbors of query in background
        # bg = find_neighbors_within_radius(query=q, hasher=self.hasher, r=self.r, dataset=self.background)
        # del bgrep
        # Neighbor counts
        n_b = list(background_counts.values()) # in background
        n_f = list(foreground_counts.values()) # in foreground
        # Hypergeometric testing
        n = self.rsize
        M = len(self.background_size) + n
        pvals = [hypergeom.sf(i, M, n, i+j) for i,j in zip(n_f,n_b)]
        seq = list(background_counts.keys())
        return pd.DataFrame({'sequence':seq, 'fg_n':n_f, 'bg_n':n_b, 'pvalue':pvals})

    def adaptive_sampling(self, fdr_threshold:float=1e-08, depth=1e6, regime=None):
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
                print(f'Sampling depth {depth} reached')
                break
            else:
                self._background(int(n))
            sign = self.neighbor_significance(nf)
            filtered = sign[(sign.bg_n == 0) | (sign.pvalue < 0.0005)]
            nf = {i:j for i,j in zip(filtered.sequence,filtered.fg_n)}
            n_elim = len(sign) - len(filtered)
            print(f'Eliminated {n_elim} TCRs,\n{len(nf)} remaining.')
        return sign