import numpy as np
import pandas as pd
import parmap
import faiss
import time
import json
import os

from scipy.stats import hypergeom
from multiprocessing import Pool, cpu_count
from typing import Union
from functools import reduce

from .hashing import Cdr3Hasher, TCRDistEncoder
from .indexing import IvfIndex, FlatIndex
from .background import match_vj_distribution, SyntheticBackground
from .constants.datasets import sample_tcrs

def above_threshold(df, row, t):
    for column in df.columns:
        if row[column] > t:
            return column
    return None

def tcr_dict_to_df(neighbor_counts, cutoff=0, add_counts=False):
    '''
    Silly helper function to convert dictionary of joined V_CDR3
    representations into a DataFrame format.
    '''
    neighbor_counts = {i:neighbor_counts[i] for i in neighbor_counts if neighbor_counts[i] >= cutoff}
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
        return dict([(j, int(lim[i+1]-lim[i])-1) for i,j in enumerate(query)])
    elif isinstance(query, pd.DataFrame):
        indices = (query['v_call'] + "_" + query['junction_aa']).to_list()
        return dict([(j, int(lim[i+1]-lim[i])-1) for i,j in enumerate(indices)])

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
    X = X.astype(np.float32)
    lim, D, I = index.idx.range_search(X, thresh=r)
    return assign_ids(query=index.hasher.tcrs, lim=lim)

class NeighborEnrichment():
    def __init__(
        self,
        repertoire:Union[pd.DataFrame, list],
        hasher:Union[Cdr3Hasher, TCRDistEncoder]=None,
        background=None,
        exact=True,
        k=None,
        n=None
        # custom_background=None, -> add this functionality back in later
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
        if hasher == None:
            self.hasher = TCRDistEncoder(aa_dim=8, full_tcr=True).fit()
        else:
            self.hasher = hasher

        self.rsize = len(repertoire)
        # self.r = radius
        self.background = background
        self.nbr_counts = None
        self.bg_index = None

        self.hasher.fit()

        if exact:
            self.fg_index = FlatIndex(hasher=self.hasher)
        else:
            if k is None:
                k = np.round(self.rsize/1000,0)
            if n is None:
                n = np.round(k/50,0)
            self.fg_index = IvfIndex(hasher=self.hasher, n_centroids=int(k), n_probe=int(n))

        # if ncpus == -1: # if set to -1, use all CPUs
        #     self.ncpus = cpu_count()
        # else:
        #     self.ncpus = ncpus

    def fixed_radius_neighbors(self, radius:Union[int,float]=12.5, exhaustive:bool=True, k=None, n=None):
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
        # if exhaustive:
        #     self.fg_index = FlatIndex(hasher=self.hasher)
        # else:
        #     if k is None:
        #         k = np.round(self.rsize/1000,0)
        #     if n is None:
        #         n = np.round(k/50,0)
        #     self.fg_index = IvfIndex(hasher=self.hasher, n_centroids=int(k), n_probe=int(n))
        self.fg_index.add(self.repertoire)
        self.r = radius
        self.nbr_counts = tcr_dict_to_df(
            index_neighbors(query=self.repertoire, index=self.fg_index, r=self.r),
            add_counts=True
        )

    def flexible_radius_neighbors(self, radii:list, t:int, exhaustive=True):
        self.fg_index.add(self.repertoire)
        all_res = []
        for i in radii:
            print(f"Compute neighbors at TCRdist > {i}")
            res = tcr_dict_to_df(index_neighbors(query=self.repertoire, index=self.fg_index, r=i), add_counts=True)
            res = res.rename(columns={"neighbors":f"nn_{i}"})
            all_res.append(res)
            merged = reduce(
                lambda left,right: pd.merge(left,right,on=['v_call','junction_aa'], how='outer'),
                all_res
                )
        print(f"Compute argmin(d,T) > {t}")
        neighbors_at_radius = merged[merged.columns[2:]]
        merged["argmin_d"] = neighbors_at_radius.apply(lambda x: above_threshold(df=neighbors_at_radius,row=x,t=t), axis=1)
        merged = merged.dropna()
        merged["neighbors"] = [i[1][i[1].argmin_d] for i in merged.iterrows()]
        merged["argmin_d"] = merged["argmin_d"].apply(lambda x: float(x.split("_")[1]))
        self.nbr_counts = merged[["v_call","junction_aa","argmin_d","neighbors"]]
                

    def _setup_background_index(self, exhaustive=True, ratio=10):
        '''

        '''
        if self.bg_index is None:
            if self.background is None:
                depth = self.repertoire.shape[0] * ratio
                print(f'Background index not set up.\nSampling background of size {depth}.')
                # bg = match_vj_distribution(n=depth, foreground=self.repertoire)
                seq_gen = SyntheticBackground(repertoire=self.repertoire,factor=ratio)
                self.background = seq_gen.shuffle_repertoire()
                print("Background contructed.")
            else:
                pass
            if exhaustive:
                self.bg_index = FlatIndex(hasher=self.hasher)
            else:
                depth = self.background.shape[0]
                k = int(depth/500)
                n = int(k/10)
                self.bg_index = IvfIndex(hasher=self.hasher, n_centroids=k, n_probe=n)
            self.bg_index.add(self.background)
            del self.background
        else:
            print(f'Using background of size {len(self.bg_index.ids)}.')

    def _prefilter(self, k):
        '''
        Filter out all foreground TCRs whose neighbor count is less than
        or equal to the background neighbor count.
        '''
        # Prepare prefilter index
        prefilter_index = FlatIndex(hasher=self.hasher)
        seq_gen = SyntheticBackground(repertoire=self.repertoire,factor=1)
        bg = seq_gen.shuffle_repertoire()
        # bg = match_vj_distribution(n=self.rsize, foreground=self.repertoire)
        prefilter_index.add(bg)
        del bg
        # Compute neighbors
        seq_with_nbrs = tcr_dict_to_df(self.nbr_counts, add_counts=True)
        nbrs_in_background = index_neighbors(query=seq_with_nbrs, index=prefilter_index, r=self.r)
        nbrs_in_background = tcr_dict_to_df(nbrs_in_background, add_counts=True)
        # Prep results
        merged = seq_with_nbrs.merge(nbrs_in_background, on=['v_call', 'junction_aa'])
        filtered = merged[merged['neighbors_x']>merged['neighbors_y']]
        filtered = filtered.drop(columns=['neighbors_y']).rename(columns={'neighbors_x':'neighbors'})
        return filtered

    def _hypergeometric(self, data):
        '''
        Compute p-values for based on foreground and background neighbor counts
        using the hypergeometric distribution.

        NOTE:
        SciPy's hypergeometric sf function uses the boost distribution implementation.
        Integer values in the boost distribution use the C unsigned integer type, which for most
        compilers is 32 bits. Any values exceedingn 32 bits will prompt either nan or strange
        p-values.
        '''
        # Hypergeometric testing
        N = self.rsize - 1
        M = len(self.bg_index.ids) + N
        if M >= 2**32:
            import warnings
            warnings.warn(f"Population size exceeds {2**32-1} (32 bits).\n p-values may be miscalculated.")
        data['pval'] = data.apply(
            lambda x: hypergeom.sf(
                x['neighbors']-1, 
                M, 
                x['neighbors'] + x['background_neighbors'], 
                N
                ),
            axis=1
            )
        return data.sort_values(by='pval')

    def foreground_neighbors_to_json(self, file):
        '''
        Dump foreground neighbor dictionary to json file.
        '''
        json_string = json.dumps(self.nbr_counts)
        f = open(file,"w")
        f.write(json_string)
        f.close()

    def foreground_neighbors_from_json(self, file):
        '''
        Load foreground neighbor dictionary from json file.
        '''
        with open(file) as json_file:
            self.nbr_counts = json.load(json_file)

    def dump_background_index_to_file(self, name):
        '''
        Write the contents of an index to disk. This function
        will create a new folder containing one binary file
        that stores the faiss index, and one file that stores
        the TCR sequence ids.

        Parameters
        ----------
        name: str
            Name of the folder to dump the files.
        '''
        assert self.bg_index is not None, "No background index found."
        os.mkdir(name)
        json_string = json.dumps(self.bg_index.ids)
        faiss.write_index(self.bg_index.idx, os.path.join(name,'index.bin'))
        f = open(os.path.join(name,'ids.json'),"w")
        f.write(json_string)
        f.close()

    def load_background_index_from_file(self, name:str, exhaustive:bool=True):
        '''
        Load a background index that is saved on disk.

        Parameters
        ----------
        name: str
            Name of the folder containing the files necessary for
            setting up an index (ids and faiss index).
        exhaustive: bool
            When True, use the exact IndexFlatL2, otherwise
            resort to the approximate IndevIVFFlat.
        '''
        faiss_index = faiss.read_index(os.path.join(name,"index.bin"))
        with open(os.path.join(name,"ids.json")) as json_file:
            index_ids = json.load(json_file)
        if exhaustive:
            self.bg_index = FlatIndex(hasher=self.hasher)
            self.bg_index.idx = faiss_index
            self.bg_index.ids = index_ids
        else:
            k = int(self.repertoire)/10
            n = int(k/10)
            self.bg_index = IvfIndex(hasher=self.hasher, n_centroids=k, n_probe=n)
            self.bg_index.idx = faiss_index
            self.bg_index.ids = index_ids

    def compute_pvalues(self, prefilter=False, q=1, ratio=10, fdr=1, exhaustive=False):
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
        depth = self.rsize * ratio
        
        # Prefilter
        if prefilter:
            print('Prefiltering repertoire.')
            filtered = self._prefilter(k=int(depth/1000))
        else:
            pass
        filtered = self.nbr_counts[self.nbr_counts.neighbors>0]
        
        # Background
        print("Setting up background index...")
        self._setup_background_index(exhaustive=exhaustive)

        # Find neighbors in background
        print(f'Computing neighbor counts in background for {len(filtered)} sequences.')
        if "argmin_d" in filtered.columns:
            neighbor_dist = {}
            for i in filtered.argmin_d.unique():
                sub = filtered[filtered.argmin_d==i]
                neighbor_dist[i] = index_neighbors(query=sub, index=self.bg_index, r=i)
            comb = []
            for i in neighbor_dist:
                res_df = tcr_dict_to_df(neighbor_dist[i], add_counts=True)
                res_df["argmin_d"] = i
                res_df = res_df.rename(columns={"neighbors":"background_neighbors"})
                comb.append(res_df)
            comb = pd.concat(comb)
            filtered = filtered.merge(comb)
        else:
            nbrs_in_background = index_neighbors(query=filtered, index=self.bg_index, r=self.r)
            nbrs_in_background = tcr_dict_to_df(nbrs_in_background, add_counts=True)
            nbrs_in_background["neighbors"] = nbrs_in_background["neighbors"] * q
            nbrs_in_background = nbrs_in_background.rename(columns={"neighbors":"background_neighbors"})
            filtered = filtered.merge(nbrs_in_background)
        # Hypergeometric testing
        print('Performing hypergeometric testing.')
        p_values = self._hypergeometric(data=filtered)
        self.significant = p_values[p_values.pval<=fdr]
        return self.significant

    
#     from functools import reduce

# np.random.seed(42)



# encoder = TCRDistEncoder().fit()
# print("Building foreground index--")
# index = FlatIndex(hasher=encoder)
# index.add(df_test)

# radii = [12.5,18.5,24.5,30.5,36.5]



# df_merged


        # col_remap = {'neighbors_x':'foreground_neighbors', 'neighbors_y':'background_neighbors'}
        # merged = filtered.merge(nbrs_in_background, on=['v_call', 'junction_aa'])
        # merged = merged.rename(columns=col_remap)

        # # Hypergeometric testing
        # M = len(self.bg_index.ids) + self.rsize
        # N = self.rsize
        # print('testing')
        # merged['pval'] = merged.apply(
        #     lambda x: hypergeom.sf(x['foreground_neighbors']-1, M, x['foreground_neighbors'] + x['background_neighbors'], N),
        #     axis=1
        #     )
        # merged = merged.sort_values(by='pval')
        # self.significant = merged[merged.pval<=fdr]
        # return self.significant

    

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