from abc import ABC
from functools import partial
from typing import List, Tuple, Callable, Union

import faiss
from pynndescent import NNDescent
import numpy as np
import pandas as pd

from .encoding import TCRDistEncoder
from .analysis import TcrCollection

from timeit import default_timer as timer


class BaseIndex(ABC):
    """
    Abstract structure for an index, supports adding TCRs and searching them.
    This index structure allows searching in two ways:
        - kNN retrieval: returns k-nearest neighbors for any given query
        - radius search: returns all TCRs within a radius r of the query
    """

    def __init__(self, idx: faiss.Index, encoder: TCRDistEncoder) -> None:
        super().__init__()
        self.idx = idx
        self.encoder = encoder
        self.ids = {}
        self.n = 0

    def _add_hashes(self, hashes):
        if not self.idx.is_trained:
            self.idx.train(hashes)
        self.idx.add(hashes)

    def _add_ids(self, X):
        if isinstance(self.encoder, TCRDistEncoder):
            if isinstance(X, pd.DataFrame):
                if self.encoder.full_tcr:
                    for x in self.encoder.tcrs.iterrows():
                        self.ids[self.n] = x[1]['v_call'] + "_" + x[1]['junction_aa']
                        self.n += 1
            else:
                for x in X:
                    self.ids[self.n] = x
                    self.n += 1
        else:
            for x in X:
                self.ids[self.n] = x
                self.n += 1

    def add(self, X: TcrCollection):
        """
        Add sequences to the index.

        Parameters
        ----------
        X : TcrCollection
            Collection of TCRs to add. Can be a Repertoire, list of Clusters, or
            list of str.
        """
        if isinstance(X, pd.DataFrame):
            if not self.encoder.full_tcr:
                import warnings
                warnings.warn(f"Provided DataFrame but 'full_tcr' was set to {self.encoder.full_tcr} --> only using CDR3 sequences to create embedding.")
        hashes = self.encoder.transform(X).astype(np.float32)
        self._add_hashes(hashes)
        self._add_ids(X)
        return self

    def _assert_trained(self):
        if not self.idx.is_trained:
            raise ValueError("Index is untrained, please add first.")

    def _search(self, x, k):
        return self.idx.search(x=x, k=k)

    def knn_search(self, y: TcrCollection, k: int = 100):
        """
        Search index for k-nearest neighbours.

        Parameters
        ----------
        y : TcrCollection
            Query TCRs.
        k : int, default = 100
            Number of nearest neighbours to search.

        Returns
        -------
        KnnResult
            `KnnResult` object.
        """
        self._assert_trained()
        hashes = self.encoder.transform(y).astype(np.float32)
        D, I = self._search(x=hashes, k=k)
        return KnnResult(y, D, I, self.ids)

    def _within_radius(self, x, r):
        if len(x.shape) == 1:
            xq = np.expand_dims(x, axis=0)
        else:
            xq = x
        return self.idx.range_search(x=xq, thresh=r)

    def _report_radius(self, I, D, exclude_self:bool=True):
        if exclude_self:
            return [(self.ids[i],d) for (i,d) in zip(I,D) if d > 0]
        else:
            return [(self.ids[i],d) for (i,d) in zip(I,D)]

    def radius_search(self, query:Union[str,list,pd.DataFrame], r:float, exclude_self=True):
        """
        ONLY COMPATIBLE WITH THE FOLLOWING INDEX STRUCTURES:
        - IndexFlat
        - IndexIVFFlat
        - IndexScalarQuantizer
        - IndexIVFScalarQuantizer
        Retrieve all items that are within a radius around the query point.

        Parameters
        ----------
        query
            TCR sequence(s) set as the centroid around which to query.
        r : float
            Radius.
        """
        q = self.encoder.transform(query).astype(np.float32)
        lims, D, I = self._within_radius(x=q, r=r)
        return self._report_radius(I, D, exclude_self=exclude_self)

    def radius_search_list(self, query, r):

            
            # nbr_df = pd.DataFrame({'target':nbrs,'distance':dist}).sort_values(by='distance')
            # nbr_df['source'] = clone_id
            # result.append(nbr_df)

        # col_order = ['source', 'target', 'distance']
        # result = pd.concat(result)
        # result = result[col_order]
        # Remove any self-sequences that are identical to the query sequence
        # if exclude_self:
            # return result[result.distance > 0]
        return result

    def array_search(self, query:pd.DataFrame, r:Union[float,int], exclude_self=False):
        xq = self.encoder.transform(query).astype(np.float32)
        lims, D, I = self.idx.range_search(x=xq, thresh=r)
        result = []
        n = 0
        for i,j in enumerate(query.index):
            nn = lims[i+1]-lims[i]
            nbr_ids = I[int(n):int(n+nn)]
            dist = D[int(n):int(n+nn)]
            n += nn
            clone_id = '_'.join(query.loc[j][['v_call', 'junction_aa']])
            nbrs = [self.ids[j] for j in nbr_ids]
            # result[clone_id] = (nbrs,dist)
            
            nbr_df = pd.DataFrame({'target':nbrs,'distance':dist}).sort_values(by='distance')
            nbr_df['source'] = clone_id
            result.append(nbr_df)

        col_order = ['source', 'target', 'distance']
        result = pd.concat(result)
        result = result[col_order]
        # Remove any self-sequences that are identical to the query sequence
        if exclude_self:
            return result[result.distance > 0]
        return result


    def dump_as_json(self, name):
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
        # assert self.bg_index is not None, "No background index found."
        # os.mkdir(name)
        json_string = json.dumps(self.idx.ids)
        faiss.write_index(self.bg_index.idx, os.path.join(name,'index.bin'))
        f = open(os.path.join(name,f'{name}.json'),"w")
        f.write(json_string)
        f.close()

class FlatIndex(BaseIndex):
    """
    Exact search for euclidean hash disstrtance.
    """

    def __init__(self, hasher: TCRDistEncoder) -> None:
        """
        Initialize index.

        Parameters
        ----------
        hasher : TCRDistEncoder
            Fitted TCRDistEncoder class.
        """
        idx = faiss.IndexFlatL2(hasher.m)
        super().__init__(idx, hasher)


class PynndescentIndex(BaseIndex):
    """
    Approximate search using PyNNDescent.
    """

    def __init__(
        self,
        hasher: TCRDistEncoder,
        k: int = 100,
        diversify_prob: float = 1.0,
        pruning_degree_multiplier: float = 1.5,
    ):
        """
        Initialize index.
        """
        idx = partial(
            NNDescent,
            n_neighbors=k,
            diversify_prob=diversify_prob,
            pruning_degree_multiplier=pruning_degree_multiplier,
        )
        idx.is_trained = True
        super().__init__(idx, hasher)


    def _add_hashes(self, X):
        self.idx = self.idx(X)
        self.idx.is_trained = True
    
    def _search(self, y, k):
        I, D = self.idx.query(y, k=k)
        return D,I

    def _search_self(self):
        I, D = self.idx.neighbor_graph
        return KnnResult(self.ids.values(), D, I, self.ids)


    def knn_search(self, y: TcrCollection=None):
        """
        Search index for nearest neighbours.

        Parameters
        ----------
        y : TcrCollection, optional
            The query TCRs. If not passed, returns the neighbours within the
            data added to the index, which is much faster.
        """
        if not y:
            return self._search_self()
        return super().knn_search(y)
    

class BaseApproximateIndex(BaseIndex):
    """
    Abstract class for approximate indexes implementing the `n_probe` property.
    """

    @property
    def n_probe(self):
        return faiss.extract_index_ivf(self.idx).nprobe

    @n_probe.setter
    def n_probe(self, n: int):
        ivf = faiss.extract_index_ivf(self.idx)
        ivf.nprobe = n


class IvfIndex(BaseApproximateIndex):
    def __init__(
        self, hasher: TCRDistEncoder, n_centroids: int = 32, n_probe: int = 5
    ) -> None:
        """
        Inverted file index for approximate nearest neighbour search.

        Parameters
        ----------
        hasher : TCRDistEncoder
            Fitted hasher object to transform CDR3 to vectors.
        n_centroids : int, default=32
            Number of centroids for the initial k-means clustering.
        n_probe : int, default=5
            Number of centroids to search at query time. Higher n_probe means
            higher recall, but slower speed.
        """
        idx = faiss.index_factory(hasher.m, f"IVF{n_centroids},Flat")
        super().__init__(idx, hasher)
        self.n_probe = n_probe


class HnswIndex:
    def __init__(self, hasher: TCRDistEncoder, n_links: int = 32) -> None:
        """
        Index based on Hierarchical Navigable Small World networks.

        Parameters
        ----------
        hasher : TCRDistEncoder
            Fitted hasher object to transform CDR3 to vectors.
        n_links : int, default=32
            Number of bi-directional links created for each element during index
            construction. Increasing M leads to better recall but higher memory
            size and slightly slower searching.

        """
        idx = faiss.index_factory(64, f"HNSW{n_links},Flat")
        super().__init__(idx, hasher)


class FastApproximateIndex(BaseApproximateIndex):
    def __init__(
        self,
        hasher: TCRDistEncoder,
        n_centroids: int = 256,
        n_links: int = 32,
        n_probe: int = 10,
    ) -> None:
        """
        Approximate index based on a combination of IVF and HNSW using scalar
        quantizer encoding.

        Parameters
        ----------
        hasher : TCRDistEncoder
            Fitted hasher object to transform CDR3 to vectors.
        n_centroids : int, default=32
            Number of centroids for the initial k-means clustering.
        n_probe : int, default=5
            Number of centroids to search at query time. Higher n_probe means
            higher recall, but slower speed.
        n_links : int, default=32
            Number of bi-directional links created for each element during index
            construction. Increasing M leads to better recall but higher memory
            size and slightly slower searching.
        """
        idx = faiss.index_factory(64, f"IVF{n_centroids}_HNSW{n_links},SQ6")
        super().__init__(idx, hasher)
        self.n_probe = n_probe


class KnnResult:
    """
    Result of k-nearest neighbor search.
    """

    def __init__(self, y, D, I, ids) -> None:
        self.D = np.sqrt(D)
        self.I = I
        self.y_idx = {y: i for i, y in enumerate(y)}
        self.ids = ids
        query_size, k = D.shape
        self.query_size = query_size
        self.k = k

    def __repr__(self) -> str:
        return f"k-nearest neighbours result (size={self.query_size}, k={self.k})"

    def _extract_neighbours(self, cdr3:str):
        try:
            i = self.y_idx[cdr3]
        except KeyError:
            raise KeyError(f"{cdr3} was not part of your query")
        I_ = np.vectorize(self._annotate_id)(self.I[i])
        return i, I_

    def extract_neighbours(self, cdr3: str) -> List[Tuple[str, float]]:
        """
        Query the KnnResult for neighbours of a specific sequence.

        Parameters:
        -----------
        cdr3 : str
            Query sequence.

        Returns
        -------
        List[(str, float)]
            List of matches, containing (sequence, score) tuples.
        """
        i, I_ = self._extract_neighbours(cdr3)
        return list(zip(I_, self.D[i]))

    def extract_neighbour_sequences(self, cdr3:str) -> List[str]:
        return self._extract_neighbours(cdr3)[1]

    def _annotate_id(self, cdr3_id):
        return self.ids.get(cdr3_id)

    def _refine_edges_iterator(self, distance_function, threshold):
        for s1 in self.y_idx.keys():
            seqs = self.extract_neighbour_sequences(s1)
            for match in ((s1, s2, dist) for s2 in seqs if (dist := distance_function(s1, s2)) <= threshold):
                yield match

    def _iter_edges(self):
        pass

    def refine(self, distance_function:Callable, threshold:float, k:int=None) -> pd.DataFrame:
        """
        Perform a second round refinement of k-nearest neighbour matches, using a custom distance function and threshold.

        Parameters:
        distance_function : Callable,
            A function taking in two string arguments, returning the distance between the sequences.
        threshold : float
            Only sequence pairs at or below this distance are retained.
        k : int, optional
            Only the k closest matches for each query sequence are retained, if below the threshold.
        """
        df = pd.DataFrame(self._refine_edges_iterator(distance_function, threshold))
        if not df.empty:
            df.columns = ["query_cdr3", "match_cdr3", "distance"]
        if k:
            df = df.sort_values('distance').groupby("query_cdr3", sort=False).head(k).sort_index()
        return df

    def as_network(self, max_edge: float = 15):
        raise NotImplementedError()