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

from .encoding import TCRDistEncoder
from .indexing import IvfIndex, FlatIndex
from .background import Background
from .constants.parsing import check_formatting

def above_threshold(df, row, t):
    for column in df.columns:
        if row[column] > t:
            return column
    return None

def tcr_dict_to_df(neighbor_counts, cutoff:int=0, add_counts:bool=True):
    '''
    Helper function to convert dictionary of joined V_CDR3
    representations into a DataFrame format.

    Parameters
    ----------
    neighbor_counts
        Dictionary of TCRs (format: {V_CDR3 : n}).
        This corresponds to the output of the radius search function in NeighborEnrichment
    cutoff, int
        Filters out TCRs with <= cutoff neighbors. Default is 0
    add_counts, bool
        Add neighbor counts. Default is True.
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
    X = index.encoder.fit_transform(query)
    X = X.astype(np.float32)
    lim, D, I = index.idx.range_search(X, thresh=r)
    return assign_ids(query=index.encoder.tcrs, lim=lim)

def neighbor_retrieval(query, encoder=None, background=None, index_type="exact", d=24.5, drop_zero_neighbor=False):
    """
    Retrieve neighboring T-cell receptor (TCR) sequences from a background set based on a similarity threshold.

    This function takes a query set of TCR sequences and a background set (or uses the query set itself if none provided),
    and retrieves neighboring TCR sequences from the background set based on their similarity to the query set,
    using a specified encoding method.

    Parameters:
        query (pandas.DataFrame): A DataFrame containing TCR sequences to be used as the query set.
        encoder (Optional[TCRDistEncoder]): An instance of TCRDistEncoder for encoding TCR sequences.
            If not provided, a default TCRDistEncoder will be used.
        background (Optional[pandas.DataFrame]): A DataFrame containing TCR sequences to be used as the background set.
            If not provided, the query set will be used as the background set.
        d (float): The similarity threshold. TCR sequences with a similarity score below this threshold will be excluded.
        drop_zero_neighbor (bool): If True, TCRs with no neighbors within the specified threshold will be excluded from the result.

    Returns:
        dict: A dictionary containing information about the neighboring TCR sequences for each query TCR.
            The keys of the dictionary are tuples representing the (v_call, junction_aa) of the query TCRs.
            The values are tuples containing:
                - The number of neighbors found within the threshold (nneigh).
                - List of indices of neighboring TCRs in the background set (idx_ids).
                - List of similarity distances between query TCRs and their neighbors (idx_dist).
                - List of neighboring TCR sequences (idx_tcr).

    Note:
        - TCRDistEncoder is used to encode TCR sequences into numerical vectors for similarity comparison.
        - The function performs range search on the encoded TCR sequences to find neighbors.
        - If drop_zero_neighbor is True, TCRs with no neighbors within the threshold are excluded from the result.

    Example:
        query_df = pd.DataFrame({'v_call': ['TRBV6-1*01', 'TRBV7-2*01'], 'junction_aa': ['CASSSPSGAGALHF', 'CASSSRDLPTEAFF']})
        background_df = pd.DataFrame({'v_call': ['TRBV3-1*01', 'TRBV3-1*01'], 'junction_aa': ['CASSQPEPTSFERANTGELFF', 'CASSQPRISTSGGNSTDTQYF']})
        neighbor_dict = neighbor_retrieval(query_df, background_df, d=24.5)
    """
    # If none provided, use default TCRDistEncoder
    if encoder == None:
        encoder = TCRDistEncoder(full_tcr=True, aa_dim=16).fit()
    # Encode query TCRs and create index
    print("encoding query sequences")
    query_vecs = encoder.transform(query)
    # If the user provides an index, we can skip the index building step
    if isinstance(background, (FlatIndex, IvfIndex)):
        assert background.n > 0, "Empty index provided. Please provide a fitted index or a data frame with TCRs."
        index = background
    else:
        # If none provided, search query TCRs against itself
        if background is None:
            background = query
        if index_type == "exact":
            print("building flat index")
            index = FlatIndex(encoder=encoder)
        else:
            print("building IVF index")
            n_centr = int(background.shape[0] / 1000)
            n_probe = int(n_centr / 50)
            index = IvfIndex(encoder=encoder, n_centroids=n_centr, n_probe=n_probe)
        index.add(background)
    # Perform search on the index
    print("searching for neighbors")
    lims, D, I = index.idx.range_search(query_vecs.astype(np.float32), thresh=d)
    # Build the neighbor dictionary
    print("Combinding results")
    neighbor_dict = {}
    for n, i in enumerate(query.iterrows()):
        nneigh = lims[n+1]-lims[n]
        idx_ids = list(I[lims[n]:lims[n+1]])
        idx_dist = list(D[lims[n]:lims[n+1]])
        idx_tcr = [index.ids[j] for j in idx_ids]
        neighbor_dict[(i[1].v_call, i[1].junction_aa)] = (nneigh, idx_ids, idx_dist, idx_tcr)
    # Drop out TCRs with no neighbors <= d if True
    if drop_zero_neighbor:
        neighbor_dict = {i:neighbor_dict[i] for i in neighbor_dict if neighbor_dict[i][0] > 0}

    return neighbor_dict

def neighbor_dict_to_df(neighbor_dict):
    """
    Convert a neighbor dictionary to a pandas DataFrame representing an adjacency list.

    This function takes a neighbor dictionary, which contains information about neighboring TCR sequences for each query TCR,
    and converts it into a pandas DataFrame that represents an adjacency list suitable for network analysis.

    Parameters:
        neighbor_dict (dict): A dictionary containing information about neighboring TCR sequences.
            The keys are tuples representing the (v_call, junction_aa) of query TCRs.
            The values are tuples containing:
                - The number of neighbors found (nneigh).
                - List of indices of neighboring TCRs (idx_ids).
                - List of similarity distances between query TCRs and their neighbors (idx_dist).
                - List of neighboring TCR sequences (idx_tcr).

    Returns:
        pandas.DataFrame: A DataFrame representing an adjacency list with columns:
            - 'idx': Index of the neighboring TCR.
            - 'tcrdist': Similarity distance between the query TCR and the neighbor.
            - 'target': TCR sequence of the neighbor.
            - 'source': TCR sequence of the query TCR (formatted as 'v_call_junction_aa').
            - 'target_v_call': V gene of the neighbor.
            - 'target_junction_aa': CDR3 of the neighbor.
            - 'source_v_call': V gene of the query TCR.
            - 'source_junction_aa': CDR3 of the query TCR.

    Example:
        neighbor_dict = {('V1', 'ACDE'): (2, [0, 1], [0.1, 0.2], ['V2_ACDF', 'V3_EFGH'])}
        adjacency_df = neighbor_dict_to_df(neighbor_dict)
    """
    adj_list = []
    # For every TCR in the neighbor dictionary, create an adjacency matrix
    for i in neighbor_dict:
        n = neighbor_dict[i][0] # number of neighbors
        df = pd.DataFrame([tuple(j[x] for j in neighbor_dict[i][1:]) for x in range(n)])
        df.columns = ["idx", "tcrdist", "target"]
        df["source"] = "_".join(i) # source TCR
        adj_list.append(df)
    # Concatenate all adjacency lists into one
    adj_list = pd.concat(adj_list)
    # Annotate V gene and CDR3
    adj_list[["target_v_call", "target_junction_aa"]] = adj_list["target"].str.split("_", expand=True)
    adj_list[["source_v_call", "source_junction_aa"]] = adj_list["source"].str.split("_", expand=True)
    
    return adj_list

class NeighborEnrichment():
    def __init__(
        self,
        repertoire:Union[pd.DataFrame, list],
        encoder:TCRDistEncoder=None,
        organism='human',
        background=None,
        exact=True,
        k=None,
        n=None,
        num_workers=1,
        add_pseudocount=False
        # custom_background=None, -> add this functionality back in later
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
        encoder: Union[TCRDistEncoder]
            Transformer function used to embed sequences into vector space.
        
        '''
        self.repertoire = repertoire

        # self.background = custom_background -> add this functionality back in later
        if encoder == None:
            self.encoder = TCRDistEncoder(aa_dim=8, full_tcr=True).fit()
        else:
            self.encoder = encoder

        # before proceeding, check whether all data fields are formatted correctly
        if encoder.organism == "human":
            check_formatting(self.repertoire)
        else:
            pass
        
        self.organism = organism
        self.rsize = len(repertoire)
        # self.r = radius
        self.background = background
        self.nbr_counts = None
        self.bg_index = None
        self.num_workers = num_workers

        self.encoder.fit()

        if exact:
            self.fg_index = FlatIndex(encoder=self.encoder)
        else:
            if k is None:
                k = np.round(self.rsize/1000,0)
            if n is None:
                n = np.round(k/30,0)
            self.fg_index = IvfIndex(encoder=self.encoder, n_centroids=int(k), n_probe=int(n))

        if add_pseudocount:
            self.pseudocount = 1
        else:
            self.pseudocount = 0


    def fixed_radius_neighbors(self, radius:Union[int,float]=12.5, k=None, n=None):
        '''
        Find the number of neighbors within a repertoire of interest.

        Parameters
        ----------
        radius: [int, float]
            Radius (in TCRdist units) around which to search for neighbors.
        k
            Number of centroids for fitting faiss.IndexIVFFlat.
        n
            Number of cells to probe when using the faiss.IndexIVFFlat.
        '''
        self.fg_index.add(self.repertoire)
        self.r = radius
        self.nbr_counts = tcr_dict_to_df(
            index_neighbors(query=self.repertoire, index=self.fg_index, r=self.r),
            add_counts=True
        )

    def flexible_radius_neighbors(self, radii:list, t:int):
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
                seq_gen = Background(repertoire=self.repertoire,factor=ratio,organism=self.organism)
                self.background = seq_gen.shuffled_background(num_workers=self.num_workers)
                print("Background contructed.")
            else:
                pass
            if exhaustive:
                self.bg_index = FlatIndex(encoder=self.encoder)
            else:
                depth = self.background.shape[0]
                k = int(depth/500)
                n = int(k/10)
                self.bg_index = IvfIndex(encoder=self.encoder, n_centroids=k, n_probe=n)
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
        prefilter_index = FlatIndex(encoder=self.encoder)
        seq_gen = Background(repertoire=self.repertoire,factor=1)
        bg = seq_gen.shuffled_background()
        # bg = match_vj_distribution(n=self.rsize, foreground=self.repertoire)
        prefilter_index.add(bg)
        del bg
        # Compute neighbors
        seq_with_nbrs = self.nbr_counts[self.nbr_counts.neighbors>0]
        indexed = index_neighbors(query=seq_with_nbrs, index=prefilter_index, r=self.r)
        indexed = tcr_dict_to_df(indexed)

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
                x['neighbors'] + x['background_neighbors'] + self.pseudocount, 
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
            self.bg_index = FlatIndex(encoder=self.encoder)
            self.bg_index.idx = faiss_index
            self.bg_index.ids = index_ids
        else:
            k = int(self.repertoire)/10
            n = int(k/10)
            self.bg_index = IvfIndex(encoder=self.encoder, n_centroids=k, n_probe=n)
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
        self._setup_background_index(exhaustive=exhaustive, ratio=ratio)

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
        # self.significant["background_neighbors"] = self.significant["background_neighbors"] / self.ratio
        # self.significant.columns = ["v_call","junction_aa","observed","expected","pval"]
        return self.significant

def to_dict(query, index):
    ndict = {}
    for i in range(len(query)):
        name = query.iloc[i].name
        q = pd.DataFrame(query.iloc[i]).T
        neighbors = index.radius_search(query=q, r=12.5)
        v = q.v_call.values[0]
        cdr3 = q.junction_aa.values[0]
        ndict[name] = tuple((v, cdr3, [i[0].split("_") + [i[1]] for i in neighbors]))
    return ndict

def shared(neighbors_a, neighbors_b):
    ovl = []
    # a vs b
    for i in neighbors_a:
        vcall = neighbors_a[i][0]
        junction_aa = neighbors_a[i][1]
        for j in neighbors_b:
            for k in neighbors_b[j][2]:
                if vcall == k[0]:
                    if junction_aa == k[1]:
                        ovl.append((i,j))
    # b vs a (reverse)
    for i in neighbors_b:
        vcall = neighbors_b[i][0]
        junction_aa = neighbors_b[i][1]
        for j in neighbors_a:
            for k in neighbors_a[j][2]:
                if vcall == k[0]:
                    if junction_aa == k[1]:
                        ovl.append((i,j))
    return list({tuple(sorted(item)) for item in ovl})

def to_neighborhood(neigh):
    return [neigh[0] + "_" + neigh[1]] + [i[0] + "_" + i[1] for i in neigh[2]]

def overlapping(neighbors_a, neighbors_b):
    result = []
    for i in neighbors_a:
        neighborhood_a = to_neighborhood(neighbors_a[i])
        for j in neighbors_b:
            neighborhood_b = to_neighborhood(neighbors_b[j])
            ovl = set(neighborhood_a).intersection(set(neighborhood_b))
            if len(ovl) > 0:
                result.append(tuple([(i,j),ovl]))
    return result