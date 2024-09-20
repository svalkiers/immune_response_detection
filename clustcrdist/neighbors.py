import numpy as np
import pandas as pd
import parmap
import faiss
import time
import json
import os
import faiss
import networkx as nx
import leidenalg as la
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import hypergeom
from typing import Union
from functools import reduce
from matplotlib.gridspec import GridSpec
from scipy.sparse import coo_matrix
from scipy.stats import hypergeom
from timeit import default_timer as timer

from .encoding import TCRDistEncoder
from .indexing import IvfIndex, FlatIndex
from .background import BackgroundModel
from .repertoire import Repertoire
from .viz import cdr3_logo
from .constants.parsing import check_formatting
from .constants.preprocessing import format_chain

# from clustcrdist.encoding import TCRDistEncoder
# from clustcrdist.constants.preprocessing import format_chain
# from clustcrdist.indexing import FlatIndex, IvfIndex
# from clustcrdist.background import BackgroundModel
# from clustcrdist.repertoire import Repertoire

def modify_edge_weights(graph, operation):
    for u, v, data in graph.edges(data=True):
        data['weight'] = operation(data['weight'])

def normalize_between_range(x, min_val, max_val):
    return ((x - min_val) / (max_val - min_val) * 50) + 5 

def normalize_to_interval(values, new_min=5, new_max=40):
    """
    Normalize a list of values to the interval [new_min, new_max].
    """
    original_min = min(values)
    original_max = max(values)
    
    if original_min == original_max:
        return [new_min] * len(values)
    
    return [new_min + (val - original_min) * (new_max - new_min) / (original_max - original_min) for val in values]

class SneTcrResult:
    '''
    Class for storing the results of a neighborhood analysis.

    Attributes:
        data (pd.DataFrame): The data containing the results of the neighborhood analysis.
        vecs (np.ndarray, optional): The vector encodings of the data. Defaults to None.

    Methods:
        __init__(self, sne, vecs=None): Initializes a SneTcrResult object.
        __repr__(self): Returns a string representation of the SneTcrResult object.
        __str__(self): Returns a string representation of the SneTcrResult object.
        __len__(self): Returns the length of the SneTcrResult object.
        __getitem__(self, n): Returns the n-th item of the SneTcrResult object.
        to_df(self): Converts the SneTcrResult object to a pandas DataFrame.
        get_vecs(self): Computes the vector encodings of the data.
        get_clusters(self, r=96, chain='AB'): Gets the different SNE clusters.

    '''

    def __init__(
            self, 
            sne,
            chain='AB',
            vecs=None
            ) -> None:
        '''
        Initializes a SneTcrResult object.

        Parameters:
            sne (pd.DataFrame): The data containing the results of the neighborhood analysis.
            vecs (np.ndarray, optional): The vector encodings of the data. Defaults to None.
        '''
        
        self.data = sne.reset_index(drop=True)
        self.chain = chain
        self.vecs = vecs
        self.is_clustered = False
        self.is_modified = False

    def __repr__(self) -> str:
        '''
        Returns a string representation of the SneTcrResult object.
        '''
        return f"SneTcrResult"
    
    def __str__(self) -> str:
        '''
        Returns a string representation of the SneTcrResult object.
        '''
        return f"SneTcrResult"

    def __len__(self) -> int:
        '''
        Returns the length of the SneTcrResult object.
        '''
        return len(self.data)
    
    def __getitem__(self, n):
        '''
        Returns the n-th item of the SneTcrResult object.
        '''
        return self.data.iloc[n]
    
    def to_df(self) -> pd.DataFrame:
        '''
        Converts the SneTcrResult object to a pandas DataFrame.

        Returns:
            pd.DataFrame: The SneTcrResult object as a pandas DataFrame.
        '''
        return self.data
    
    def get_vecs(self):
        '''
        Computes the vector encodings of the data.
        '''
        encoder = TCRDistEncoder(aa_dim=16, chain=self.chain).fit()
        self.vecs = encoder.transform(self.data)

    def _index_based_graph(self, r, significant=True, weighted=True):

        if significant:
            sign = self.data[self.data['evalue'] < .05]
        else:
            sign = self.data

        sign_vecs = self.vecs[sign.index]
        index_remap = {n: i for n, i in enumerate(sign.index)}

        index = faiss.IndexFlatL2(self.vecs.shape[1])
        index.add(self.vecs)

        lims, D, I = index.range_search(sign_vecs, thresh=r)
        edges = convert_range_search_output(lims, D, I)
        edges = [((index_remap[u], v), w) for (u, v), w in edges]
        if weighted:
            edges = [i[0] + tuple([i[1]]) for i in edges if len(set(i[0])) > 1]
        else:
            edges = [i[0] for i in edges if len(set(i[0])) > 1]
        nodes = set(list(sign.index) + list(I))

        return edges, nodes
    
    def _matrix_based_graph(self, r, significant=True):

        if significant:
            sign = self.data[self.data['evalue'] < .05]
        else:
            sign = self.data

        dm = compute_sparse_distance_matrix(
            tcrs=sign, 
            chain=self.chain, 
            organism='human', 
            d=r,
            m=16,
            vecs=self.vecs[sign.index]
            )
        
        non_zero_values = dm.data
        row_indices, col_indices = dm.nonzero()

        mask = (non_zero_values >= 0)
        filtered_row_indices = row_indices[mask]
        filtered_col_indices = col_indices[mask]
        ids = np.column_stack((filtered_row_indices, filtered_col_indices))

        nodes = sign.reset_index(drop=True).index.values
        edges = [(int(i[0]),int(i[1]),i[2]) for i in ids]
        
        return edges, nodes

    def get_clusters(self, r=96, significant=True, periphery=False):
        '''
        Gets the different SNE clusters.
        Builds a network from sequence neighbors (< r) and partitions
        the graph using Leiden clustering.

        Suggested thresholds for r:
            - paired chain ('AB'): 96
            - single chain ('A' or 'B'): 24

        Parameters:
            r (int): 96
                Threshold for defining a neighbor.
            chain (str): 'AB'
                TCR chain.

        Returns:
            clusters (dict): Dictionary with cluster assignments per TCR. The keys correspond to the TCR indices in the neighbor analysis dataframe.

        Also adds a cluster column to the dataframe.
        '''

        if len(self.chain) == 2:
            self.data = self.data.loc[self.data.groupby('tcr_index')['radius'].idxmin()].reset_index(drop=True)
        
        if self.vecs is None:
            self.get_vecs()

        # if significant:
        #     sign = self.data[self.data['evalue'] < .05]

        # dm = compute_sparse_distance_matrix(
        #     tcrs=sign, 
        #     chain=self.chain, 
        #     organism='human', 
        #     d=r,
        #     m=16,
        #     vecs=self.vecs[sign.index]
        #     )
        
        # non_zero_values = dm.data
        # row_indices, col_indices = dm.nonzero()

        # mask = (non_zero_values >= 0)
        # filtered_row_indices = row_indices[mask]
        # filtered_col_indices = col_indices[mask]
        # ids = np.column_stack((filtered_row_indices, filtered_col_indices))

        # dm = squareform(pdist(self.vecs, metric='euclidean')**2)
        # np.fill_diagonal(dm, -1)
        # ids = np.argwhere((dm <= r) & (dm >= 0))

        if periphery:
            self.edges, self.nodes = self._index_based_graph(r, significant=significant)
        else:
            self.edges, self.nodes = self._matrix_based_graph(r, significant=significant)
        
        # nodes = sign.reset_index(drop=True).index.values
        # edges = [(nodes[i[0]], nodes[i[1]]) for i in ids]
        self.G = nx.Graph()
        self.G.add_nodes_from(list(self.nodes))
        self.G.add_weighted_edges_from(self.edges)

        if isinstance(self.G, (nx.Graph, nx.DiGraph)):
            self.Gi = ig.Graph.from_networkx(self.G)
        partition = la.find_partition(self.Gi, la.ModularityVertexPartition)
        cluster_lists = [[list(self.nodes)[node] for node in c] for c in list(partition)]
        clusters = {int(j): n for n, i in enumerate(cluster_lists) for j in i}
        self.data['cluster'] = self.data.index.map(clusters)
        self.data['cluster'] = self.data['cluster'].fillna(-1).astype(int)

        self.is_clustered = True

        return clusters

    def draw_cluster(self, cluster_id, r=12.5, node_size=None, labels=False):
        
        assert self.is_clustered, 'Please run get_clusters() first.'

        cluster = self.data[self.data.cluster == cluster_id]

        fig = plt.figure(figsize=(4,4),dpi=150)

        if len(self.chain) == 1:

            gs = GridSpec(10,10)
            main = fig.add_subplot(gs[:8,:])
            logo_v = fig.add_subplot(gs[8:,:2])
            logo_j = fig.add_subplot(gs[8:,8:])
            logo_cdr3 = fig.add_subplot(gs[8:,2:8])

            v = cluster.v_call.value_counts()
            xv = v.index
            yv = v.values
            logo_v.bar(x=xv, height=yv, edgecolor='black')
            logo_v.set_xticklabels(xv, rotation=45, ha='right')

            j = cluster.j_call.value_counts()
            xj = j.index
            yj = j.values
            logo_j.bar(x=xj, height=yj, edgecolor='black')
            logo_j.set_xticklabels(xj, rotation=45, ha='right')
            logo_j.yaxis.set_label_position("right")
            logo_j.yaxis.tick_right()

            pos_matrix = cdr3_logo(cluster, ax=logo_cdr3)
            logo_cdr3.set_xticks([])
            logo_cdr3.set_yticks([])
        
        elif len(self.chain) == 2:

            gs = GridSpec(15,10)
            main = fig.add_subplot(gs[:8,:])
            vja = fig.add_subplot(gs[9:11,7:])
            logo_cdr3a = fig.add_subplot(gs[9:11,:5])
            vjb = fig.add_subplot(gs[13:,7:])
            logo_cdr3b = fig.add_subplot(gs[13:,:5])

            pivot_table_a = cluster[['va','ja']].pivot_table(index='va', columns='ja', aggfunc='size', fill_value=0)
            pivot_table_b = cluster[['vb','jb']].pivot_table(index='vb', columns='jb', aggfunc='size', fill_value=0)

            # ALPHA
            sns.heatmap(pivot_table_a, annot=True, fmt='d', cmap='Blues', ax=vja, annot_kws={"fontsize":5})
            xlabels = list(pivot_table_a.columns)
            vja.set_xticks([i+.5 for i in range(len(xlabels))])
            vja.set_xticklabels(xlabels, fontsize=5, rotation=45)
            ylabels = list(pivot_table_a.index)
            vja.set_yticks([i+.5 for i in range(len(ylabels))])
            vja.set_yticklabels(ylabels, fontsize=5, rotation=45)
            vja.set_xlabel('')
            vja.set_ylabel('')

            cluster_a = cluster[['va','ja','cdr3a','cdr3a_nucseq']].rename(
                columns={'va':'v_call','ja':'j_call','cdr3a':'junction_aa','cdr3a_nucseq':'junction'}
                )
            pos_matrix = cdr3_logo(cluster_a, ax=logo_cdr3a)
            logo_cdr3a.set_xticks([])
            logo_cdr3a.set_yticks([])
            logo_cdr3a.set_title(r'CDR3$\alpha$', fontsize=8)

            # BETA
            sns.heatmap(pivot_table_b, annot=True, fmt='d', cmap='Blues', ax=vjb, annot_kws={"fontsize":5})
            xlabels = list(pivot_table_b.columns)
            vjb.set_xticks([i+.5 for i in range(len(xlabels))])
            vjb.set_xticklabels(xlabels, fontsize=5, rotation=45)
            ylabels = list(pivot_table_b.index)
            vjb.set_yticks([i+.5 for i in range(len(ylabels))])
            vjb.set_yticklabels(ylabels, fontsize=5, rotation=45)
            vjb.set_xlabel('')
            vjb.set_ylabel('')

            cluster_b = cluster[['vb','jb','cdr3b','cdr3b_nucseq']].rename(
                columns={'vb':'v_call','jb':'j_call','cdr3b':'junction_aa','cdr3b_nucseq':'junction'}
                )
            pos_matrix = cdr3_logo(cluster_b, ax=logo_cdr3b)
            logo_cdr3b.set_xticks([])
            logo_cdr3b.set_yticks([])
            logo_cdr3b.set_title(r'CDR3$\beta$', fontsize=8)


        if self.is_modified:
            pass
        else:
            modify_edge_weights(self.G, lambda x: normalize_between_range(r - x, 0, r))
            self.is_modified = True

        newnodes = list(cluster.index)
        newG = self.G.subgraph(newnodes)

        pos = nx.spring_layout(newG, weight='weight')
        coordinates = np.array(list(pos.values()))
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        c = -np.log10(cluster.loc[list(newG.nodes())].evalue)
        if len(self.chain) == 1:
            cdr3 = cluster.loc[list(newG.nodes())].junction_aa
        elif len(self.chain) == 2:
            cdr3a = cluster.loc[list(newG.nodes())].cdr3a
            cdr3b = cluster.loc[list(newG.nodes())].cdr3b
            cdr3 = cdr3a + '_' + cdr3b

        # def normalize(x, min_val, max_val):
        #     return ((x - min_val) / (max_val - min_val) * 50) + 5 
        if node_size is None:
            s = 15
        elif isinstance(node_size, str):
            s = np.sqrt(cluster.loc[list(newG.nodes())][node_size])
            s = normalize_between_range(s, s.min(), s.max())
        else:
            s = node_size

        nx.set_node_attributes(newG, dict(zip(newG.nodes(), cdr3)), 'labels')
        nx.draw_networkx_edges(newG, pos, alpha=0.5, width=.5, ax=main)
        if labels:
            nx.draw_networkx_labels(newG, pos=pos, labels=nx.get_node_attributes(newG, 'labels'), font_size=4, ax=main)
        cbar = main.scatter(x,y,s=s,linewidths=0.5,edgecolor='black',alpha=1,c=c,cmap='viridis')

        plt.colorbar(cbar, ax=main, label='-log10(e-value)')

        return fig

    def draw_neighborhoods(self, ax=None, node_size=None, annotate=True):

        layout = self.Gi.layout("graphopt", niter=1000)

        if ax is None:
            fig, ax = plt.subplots(dpi=150, figsize=(5,5))
        else:
            pass

        visual_style = {'edge_width': 0.5,}
        ig.plot(self.Gi,
                layout=layout,
                **visual_style,
                target=ax)

        x = [i[0] for i in layout]
        y = [i[1] for i in layout]
        node_data = self.data.iloc[[i['_nx_name'] for i in self.Gi.vs]]
        c = -np.log10(node_data.evalue)
        if node_size is None:
            s = 15
        elif isinstance(node_size, str):
            s = normalize_to_interval(node_data[node_size])
        else:
            s = node_size

        cbar = ax.scatter(x, y, c=c, cmap='viridis', s=s, edgecolors='black', linewidths=0.5)
        plt.colorbar(cbar, ax=ax, label='-log10(e-value)')

        node_data['x_coord'] = x
        node_data['y_coord'] = y

        if annotate:
            for clus in node_data.cluster.unique():
                cluster = node_data[node_data.cluster==clus]
                ax.text(
                    x=cluster.x_coord.mean(), 
                    y=cluster.y_coord.mean(),
                    s=str(clus),
                    c='red'
                )

    # def draw_neighborhoods(self, ax=None, node_size=None, r=12.5):

    #     significant_clusters = self.data[self.data['evalue'] < 0.05]['cluster'].unique()
    #     indices = self.data[self.data['cluster'].isin(significant_clusters)].index

    #     if self.is_modified:
    #         pass
    #     else:
    #         modify_edge_weights(self.G, lambda x: normalize_between_range(r - x, 0, r))
    #         self.is_modified = True

    #     newnodes = list(indices)
    #     newedges = self.G.edges(indices)
    #     newG = self.G.subgraph(newnodes)

    #     pos = nx.spring_layout(newG, weight='weight', iterations=35)
    #     coordinates = np.array(list(pos.values()))
    #     x = coordinates[:, 0]
    #     y = coordinates[:, 1]
    #     c = -np.log10(self.data.loc[newnodes].evalue)

    #     if node_size is None:
    #         s = 15
    #     elif isinstance(node_size, str):
    #         s = np.sqrt(self.data.loc[newnodes][node_size])
    #         s = normalize_between_range(s, s.min(), s.max())
    #     else:
    #         s = node_size

    #     if ax is None:
    #         nx.draw_networkx_edges(newG, pos, alpha=0.5, width=.5)
    #         cbar = plt.scatter(x,y,s=s,linewidths=0.5,edgecolor='black',alpha=1,c=c,cmap='viridis')
    #     else:
    #         nx.draw_networkx_edges(newG, pos, ax=ax, alpha=0.5, width=.5)
    #         cbar = ax.scatter(x,y,s=s,linewidths=0.5,edgecolor='black',alpha=1,c=c,cmap='viridis')

    #     plt.colorbar(cbar, ax=ax, label='-log10(e-value)')

def neighbor_analysis(tcrs, chain: str, organism: str, radius: Union[float,int], vecs=None, background=None, encoder=None, fgindex=None, bgindex=None, depth: int=10):
    '''
    Perform a neighborhood analysis on a set of TCRs. TCR neighborhood analysis compares the
    distribution of sequence neighbors within a fixed distance radius against a background dataset.
    Neighborhood enrichment is determined by comparing the neighbor count of a TCR in the sample
    compared to the neighbor count of the same TCR in the background dataset.

    Suggestions for the choice of 'radius':
        - single chain: 12.5 (strict), 24.5 (flexible)
        - paired chain: 96.5

    Parameters:
        tcrs
            The input data provided as a pandas.DataFrame.
        chain (str): 'AB'
            TCR chain.
        organism (str): human
            Species from which the data originates.
        radius (float or int)
             Threshold for defining a neighbor.
        vecs: None
            Pre-computed vecTCRdist encodings.
        background: None
            Custom background.
        encoder : None
            Custom encoder. Accepts TCRDistEncoder only.
        fgindex: None
            Custom index for foreground TCRs.
        bgindex: None
            Custom index for background TCRs.
        depth (int): 10
            Depth of the background data set. This is a factor of the size of the input data.

    Returns:
        SneTcrResult
    '''

    chain = format_chain(chain)

    if len(chain) == 1:

        if vecs is None:
            if encoder is None:
                encoder = TCRDistEncoder(aa_dim=8,organism=organism,chain=chain).fit()
            print('Encoding TCRs.')
            vecs = encoder.transform(tcrs)

        print(f'Computing neighbor distribution for {vecs.shape[0]} TCRs.')
        if fgindex is None:
            fgindex = faiss.IndexFlatL2(vecs.shape[1])
            fgindex.add(vecs)
        
        lims, D, I = fgindex.range_search(vecs, radius)
        nbrcounts = neighbor_distr_from_lims(lims)
        nonzero = np.where(nbrcounts > 0)[0]
        del fgindex

        if background is not None:
            print('Background data provided, skipping background generation.')
        else:
            print(f'Creating background data set with {tcrs.shape[0]} TCRs.')
            background = BackgroundModel(tcrs, depth)
            bgdata = background.shuffle(chain=chain)
        
        bgvecs = encoder.transform(bgdata)

        if bgindex is None:
            bgindex = faiss.IndexFlatL2(vecs.shape[1])
            bgindex.add(bgvecs)

        print('Estimating expected neighbor distribution.')
        lims, D, I = bgindex.range_search(vecs[nonzero], radius)
        del bgindex

        estimated = neighbor_distr_from_lims(lims)

        presne = tcrs.loc[nonzero]
        presne['foreground_neighbors'] = nbrcounts[nonzero]
        presne['background_neighbors'] = estimated

        sne = _hypergeometric(presne, tcrs.shape[0], bgdata.shape[0])

    elif len(chain) > 1:

        # Make sure the data is in the paired format
        airr_cols = ['v_call','j_call','junction_aa','junction']
        if all([i in tcrs.columns for i in airr_cols]):
            rep = Repertoire(tcrs)
            tcrs = rep.airr_to_tcrdist_paired()
        else:
            pass
        
        print('Encoding TCRs')
        if vecs is None:
            if encoder is None:
                encoder = TCRDistEncoder(aa_dim=8,organism=organism,chain=chain).fit()
            vecs = encoder.transform(tcrs)
        
        # Get the neighbor distribution in the sample
        print(f'Computing neighbor distribution for {vecs.shape[0]} TCRs in sample.')
        fg_results = get_foreground_nbr_counts(vecs, radius)

        if background is None:
            # Create a background data set
            print(f'Creating background data set with {tcrs.shape[0]*depth} TCRs')
            bgmodel = BackgroundModel(repertoire=tcrs, factor=depth)
            background = bgmodel.shuffle(chain=chain)

        # Get the encodings for both chains separately in foreground and background
        print('Estimating expected neighbor distribution.')
        avecs, bvecs = encoder.transform(tcrs, split_ab=True)
        avecsbg, bvecsbg = encoder.transform(background, split_ab=True)
        bg_ab_counts = get_background_nbr_counts(avecs,bvecs,avecsbg,bvecsbg,radius)

        # Combine all results and compute the neighborhood p-values
        sne = compute_neighborhood_pvalues(tcrs, fg_results, bg_ab_counts, background.shape[0])

    return SneTcrResult(sne=sne, chain=chain)

def neighbor_distr_from_lims(lims):
    '''
    Extract the number of neighbors for each query from lims.
    '''
    num_queries = len(lims) - 1
    result = np.zeros(num_queries, dtype=int)
    for i in range(num_queries):
        start = lims[i]
        end = lims[i + 1]
        result[i] = end - start
    return result

def convert_range_search_output(lims, D, I, offset=0):
    '''
    Convert the output of a range search to a list of tuples.
    '''
    # Calculate the number of queries
    num_queries = len(lims) - 1
    # Vectorize the construction of result pairs
    result_indices = []
    result_distances = []
    for i in range(num_queries):
        start = lims[i]
        end = lims[i + 1]
        query_indices = np.full(end - start, i)
        neighbor_indices = I[start:end] + offset
        distances = D[start:end]
        result_indices.extend(zip(query_indices, neighbor_indices))
        result_distances.extend(distances)
    result = list(zip(result_indices, result_distances))
    return result
    
def compute_sparse_distance_matrix(tcrs, chain, organism, exact=True, d=96.5, m=8, encoder=None, vecs=None):
    '''
    Compute the sparse distance matrix for a set of TCRs.
    '''
    
    t0 = timer()

    if encoder is None:
        encoder = TCRDistEncoder(aa_dim=m,organism=organism,chain=chain).fit()

    if vecs is None:
        # Encode the TCRs
        start = timer()
        vecs = encoder.transform(tcrs).astype(np.float32)
        print(f'Encoding TCRs took {timer()-start:.2f}s')

    if exact:
        # Flat index will ensure 'exact' search
        # However, this is still an approximation of the true TCRdist
        start = timer()
        idx = faiss.IndexFlatL2(vecs.shape[1])
        idx.add(vecs)
        print(f'Building the index took {timer()-start:.2f}s')
    else:
        # IVF index
        start = timer()
        k = round(len(vecs)/1000)
        n = round(k/20)+2
        if n > k:
            n = k
        idx = faiss.index_factory(encoder.m, f"IVF{k},Flat")
        if vecs.shape[0] > 10000:
            num_training_samples = int(0.2 * vecs.shape[0])  # Use 20% of vecs for training
            training_vecs = vecs[np.random.choice(vecs.shape[0], num_training_samples, replace=False)]
            idx.train(training_vecs)
        else:
            idx.train(vecs)
        idx.nprobe = n
        idx.add(vecs)
        # index = IvfIndex(encoder=encoder, n_centroids=k, n_probe=n)
        # index.add(tcrs, vecs)
        print(f'Building the index took {timer()-start:.2f}s')

    # Search index
    start = timer()
    lims, D, I = idx.range_search(vecs, thresh=d)
    print(f'Searching index within radius {d} took {timer()-start:.2f}s')

    # Format the results.
    start = timer() 
    res = convert_range_search_output(lims, D, I)

    # Determine the size of the matrix
    max_index = len(tcrs)
    base = [([i,i],0) for i in range(max_index)]
    res += base
    res = list(set((tuple(indices), distance-1) for indices, distance in res))

    # Initialize lists to hold row indices, column indices, and data values
    rows, cols, values = [], [], []

    # Populate the lists with the given distances
    for indices, distance in res:
        i, j = indices
        rows.append(i)
        cols.append(j)
        values.append(distance)
        if i != j:  # Ensure the matrix is symmetric
            rows.append(j)
            cols.append(i)
            values.append(distance)

    # Convert the lists to a sparse matrix using coo_matrix
    dm = coo_matrix((values, (rows, cols)), shape=(max_index, max_index), dtype=np.float64)
    print(f'Building the sparse matrix took {timer()-start:.2f}s')
    print(f'Total time to compute distance matrix: {timer()-t0:.2f}s')

    return dm

def find_neighbors(tcrs, chain, radius, organism='human', vecs=None, encoder:TCRDistEncoder=None, index=None):
    '''
    Suggested radii:
        - single chain: 24.5
        - paired chain: 96.5
    '''

    if vecs is None:
        if encoder is None:
            encoder = TCRDistEncoder(aa_dim=8,organism=organism,chain=chain).fit()
        vecs = encoder.transform(tcrs)

    if index is None:
        index = faiss.IndexFlatL2(vecs.shape[1])
        index.add(vecs)
    
    lims, D, I = index.range_search(vecs,radius)

    return neighbor_distr_from_lims(lims)

def _hypergeometric(
        data,
        fg_size,
        bg_size,
        pseudocount=0
        ):
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
    N = fg_size - 1
    M = bg_size + N
    if M >= 2**32:
        import warnings
        warnings.warn(f"Population size exceeds {2**32-1} (32 bits).\n p-values may be miscalculated.")
    data['pvalue'] = data.apply(
        lambda x: hypergeom.sf(
            x['foreground_neighbors']-1, 
            M, 
            x['foreground_neighbors'] + x['background_neighbors'] + pseudocount, 
            N
            ),
        axis=1
        )
    data['evalue'] = data['pvalue'] * fg_size
    
    return data.sort_values(by='evalue')

def get_foreground_nbr_counts(
        fg_vecs,
        radius=96.5,
):
    '''
    Get the faiss data (lims,D,I) for range_search of tcrs against self

    so it will include self-distances

    tcrs and bg_tcrs should already have been filtered
    '''
    qvecs = fg_vecs # could optionally split into batches

    print('start IndexFlatL2 range search', qvecs.shape, fg_vecs.shape)
    start = timer()
    idx = faiss.IndexFlatL2(fg_vecs.shape[1])
    idx.add(fg_vecs)
    print(type(qvecs), type(radius))
    lims, D, I = idx.range_search(qvecs, radius)
    print(f'IndexFlatL2 range search took {timer()-start:.2f}')
    return {'lims':lims, 'D':D, 'I':I}

def get_background_nbr_counts(
        fg_avecs,
        fg_bvecs,
        bg_avecs,
        bg_bvecs,
        maxdist=96,
):
    ''' Compute the background paired tcrdist distribution by taking the
    convolution of the alpha and beta single-chain tcrdist distributions.
    The effective number of paired background comparisons is len(bg_tcrs)**2

    returns an integer-valued numpy array of shape (num_fg_tcrs, maxdist+1)

    histogram bin-size is 1.0

    first bin is (-0.5, 0.5), last bin is (maxdist-0.5, maxdist+0.5)

    tcrs and bg_tcrs should already have been filtered

    (see phil_functions.compute_background_paired_tcrdist_distributions)
    '''

    start = timer()
    ab_counts = compute_background_paired_tcrdist_distributions(
        fg_avecs, fg_bvecs, bg_avecs, bg_bvecs, maxdist)
    print(f'paired distribution calc took {timer()-start:.2f}')

    return ab_counts

def compute_background_single_tcrdist_distributions(
        fg_vecs,
        bg_vecs,
        maxdist,
        rowstep = 5000,
        colstep = 50000,
):
    ''' Compute the histogram of distances to bg_vecs for each vector in fg_vecs

    returns an integer-valued numpy array of shape (num_fg_tcrs, maxdist+1)

    histogram bin-size is 1.0

    first bin is (-0.5, 0.5), last bin is (maxdist-0.5, maxdist+0.5)

    for alphabeta tcr clumping, I use maxdist=96
    '''
    dim = fg_vecs.shape[1]
    assert dim == bg_vecs.shape[1]
    num_fg = fg_vecs.shape[0]
    num_bg = bg_vecs.shape[0]
    maxdist = int(maxdist+0.1) # confirm int
    print('compute_background_single_tcrdist_distributions: '
          f'dim= {dim} maxdist= {maxdist}')
    #maxdist_float = maxdist + 0.5
    dist_counts = np.zeros((num_fg, maxdist+1), dtype=int)

    nrows = (num_fg-1)//rowstep + 1
    ncols = (num_bg-1)//colstep + 1

    # initialize distance storage
    dists = np.zeros((rowstep*colstep,), dtype=np.float32) # 1D array

    start0 = timer()
    for ii in range(nrows):
        ii_start = ii*rowstep
        for jj in range(ncols):
            jj_start = jj*colstep
            xq = fg_vecs[ii_start:ii_start+rowstep] # faiss terminology here
            xb = bg_vecs[jj_start:jj_start+colstep]
            nq = xq.shape[0] # Num Query
            nb = xb.shape[0] # Num dataBase (?)

            start = timer()
            faiss.pairwise_L2sqr(dim, nq, faiss.swig_ptr(xq), nb, faiss.swig_ptr(xb),
                                 faiss.swig_ptr(dists))
            disttime = timer()-start
            #print(disttime)
            start += disttime

            # now fill counts
            for iq in range(nq):
                dist_counts[ii_start+iq,:] += np.histogram(
                    dists[iq*nb:(iq+1)*nb], bins = maxdist+1,
                    range = (-0.5, maxdist+0.5))[0]

            counttime = timer()-start

            #print(f'ij {ii} {jj} {disttime:.2f} {counttime:.2f} {timer()-start0:.2f}')

    return dist_counts



def compute_background_paired_tcrdist_distributions(
        fg_avecs,
        fg_bvecs,
        bg_avecs,
        bg_bvecs,
        maxdist,
        rowstep = 5000,
        colstep = 50000,
):
    ''' Compute the background paired tcrdist distribution by taking the
    convolution of the alpha and beta single-chain tcrdist distributions

    returns an integer-valued numpy array of shape (num_fg_tcrs, maxdist+1)

    histogram bin-size is 1.0

    first bin is (-0.5, 0.5), last bin is (maxdist-0.5, maxdist+0.5)
    '''

    num_fg_tcrs = fg_avecs.shape[0]
    num_bg_tcrs = bg_avecs.shape[0]
    assert num_fg_tcrs == fg_bvecs.shape[0]
    assert num_bg_tcrs == bg_bvecs.shape[0]

    acounts = compute_background_single_tcrdist_distributions(
        fg_avecs, bg_avecs, maxdist, rowstep=rowstep, colstep=colstep)

    bcounts = compute_background_single_tcrdist_distributions(
        fg_bvecs, bg_bvecs, maxdist, rowstep=rowstep, colstep=colstep)

    abcounts = np.zeros((num_fg_tcrs, int(maxdist+1)), dtype=int)

    assert acounts.shape == bcounts.shape == abcounts.shape

    for d in range(int(maxdist+1)):
        for adist in range(d+1):
            abcounts[:,d] += acounts[:,adist] * bcounts[:,d-adist]

    return abcounts

def compute_neighborhood_pvalues(
        tcrs,
        fg_results,
        bg_ab_counts,
        num_bg_tcrs,
        radii = [24, 48, 72, 96],
        min_nbrs = 2, # tcr_clumping uses 1?
        pseudocount = 0.25,
        evalue_threshold = 1,
):
    ''' Compute pvalues and simple-bonferroni corrected "evalues"
    for observed foreground neighbor numbers

    Right now this is using poisson but we could shift to hypergeometric
    I think for the paired setting where the effective number of background comparisons
    is very large the two should give pretty similar results
    '''
    from scipy.stats import poisson

    # get background counts at radii
    maxdist = max(radii)
    assert bg_ab_counts.shape[1] == maxdist+1
    bg_counts = np.cumsum(bg_ab_counts, axis=1)[:, radii]

    # get foreground counts at radii
    num_fg_tcrs = tcrs.shape[0]
    lims = fg_results['lims']
    D = fg_results['D']
    assert num_fg_tcrs == lims.shape[0]-1
    fg_counts = np.zeros((num_fg_tcrs, len(radii)), dtype=int)
    for ii, r in enumerate(radii):
        fg_counts[:,ii] = np.add.reduceat((D<=r+.5), lims[:-1].astype(int))

    fg_counts -= 1 # exclude self nbrs

    assert fg_counts.shape == bg_counts.shape == (num_fg_tcrs, len(radii))

    # "rates" are the expected number of neighbors based on the background counts
    # we divide background counts by num_bg_tcrs**2 (since that's the effective number
    # of background paired comparisons) to get the probability of seeing a neighbor
    # at a given radius, then we multiply by num_fg_tcrs to get the expected number
    # of neighbors.
    # rates.shape: (num_fg_tcrs, len(radii))
    #
    rates = (np.maximum(bg_counts, pseudocount) *
             (num_fg_tcrs/(num_bg_tcrs*num_bg_tcrs)))

    dfl = []
    for ii, (counts,rates) in enumerate(zip(fg_counts, rates)):
        for jj, (count,rate) in enumerate(zip(counts,rates)):
            if count >= min_nbrs:
                pvalue = poisson.sf(count-1, rate)
                evalue = pvalue*(len(radii)*num_fg_tcrs)
                if evalue <= evalue_threshold:
                    dfl.append(dict(
                        pvalue=pvalue,
                        evalue=evalue,
                        tcr_index=ii,
                        radius=radii[jj],
                        num_nbrs=count,
                        expected_num_nbrs=rate,
                        bg_nbrs=rate*num_bg_tcrs*num_bg_tcrs/num_fg_tcrs,
                    ))

    results = pd.DataFrame(dfl)
    if results.shape[0]:
        results = results.join(tcrs, on='tcr_index').sort_values('evalue')

    return results

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
    print("Combining results")
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
        df.columns = ["idx", "tcrdist", "background"]
        df["query"] = "_".join(i) # source TCR
        adj_list.append(df)
    # Concatenate all adjacency lists into one
    adj_list = pd.concat(adj_list)
    # Annotate V gene and CDR3
    adj_list[["background_v_call", "background_junction_aa"]] = adj_list["background"].str.split("_", expand=True)
    adj_list[["query_v_call", "query_junction_aa"]] = adj_list["query"].str.split("_", expand=True)
    
    return adj_list




# LEGACY
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


    def fixed_radius_neighbors(self, radius:Union[int,float]=12.5):
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

    def save_background_index_to_file(self, name):
        '''
        Write the contents of an index to disk. This function
        will create a new folder containing one binary file
        that stores the faiss index, and one file that stores
        the TCR sequence ids.

        Parameters
        ----------
        name: str
            Name of the folder to save the files.
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