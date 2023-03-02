import numpy as np
import pandas as pd

from typing import Union
from functools import lru_cache
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import MDS

from .analysis import TcrCollection, Repertoire, Cluster, ClusteredRepertoire
from .constants.base import AALPHABET, GAPCHAR
from .constants.hashing import DEFAULT_DM, TCRDIST_DM
from .constants.preprocessing import setup_gene_cdr_strings

from sklearn.utils.validation import check_is_fitted

def positional_encoding(sequence_len:int, m:int, p=3) -> np.ndarray:
    """
    Generate positional encoding based on sinus and cosinus functions with
    geometrically increasing wavelenghts.

    Parameters
    ----------
    sequence_len : int
        Number of AAs in the CDR3 sequence.
    m : int
        Number of output vector dimensions.
    p : int
        Exponent.
    """
    distances = np.tile(np.arange(sequence_len)/sequence_len, (m,1)).T - np.tile(np.arange(m)/m, (sequence_len,1))
    cos_distances = np.cos(distances*np.pi*2)
    pow_cos_distances = np.power(cos_distances, p)
    return pow_cos_distances

class Cdr3Hasher(BaseEstimator, TransformerMixin):
    def __init__(self, distance_matrix:np.ndarray=DEFAULT_DM, m:int=32, p:float=9, trim_left:int=0, trim_right:int=0, scale:bool=True) -> None:
        """
        Locality-sensitive hashing for amino acid sequences. Hashes CDR3
        sequences of varying lengths into m-dimensional vectors, preserving
        similarity.setup_gene_cdr_strings

        Parameters
        ----------
        distance_matrix : np.ndarray[20,20]
            Twenty by twenty matrix containing AA distances. Amino acids are ordered alphabetically  (see AALPHABET constant)
        m : int
            Number of embedding dimensions.
        p : int
            Positional importance scaling factor.
        trim_left : int
            Number of amino acids trimmed from left side of sequence.
        trim_right : int
            Number of amino acids trimmed from right side of sequence.
        """
        self.distance_matrix = distance_matrix
        self.m = m
        self.p = p
        self.trim_left = trim_left
        self.trim_right = trim_right
        self.scale = scale

    def __repr__(self):
        return f'Cdr3Hasher(m={self.m})'

    def fit(self, X=None, y=None):
        """
        Fit Cdr3Hasher. This method generates the necessary components for hashing.
        """
        self.aa_vectors_ = self._construct_aa_vectors()
        self.position_vectors_ = self._construct_position_vectors()
        return self

    def _construct_aa_vectors(self) -> dict:
        """
        Create dictionary containing AA's and their corresponding hashes, of
        type str : np.ndarray[m].
        """
        if self.scale:
            vecs = MDS(n_components=self.m, dissimilarity="precomputed", random_state=11, normalized_stress=False).fit_transform(self.distance_matrix)
        else:
            vecs = self.distance_matrix
        vecs_dict = {aa:vec for aa,vec in zip(AALPHABET, vecs)}
        return vecs_dict

    def _construct_position_vectors(self) -> dict:
        """
        Create positional encoding matrix for different CDR3 lengths

        Returns
        -------
        dict int : np.ndarray[m, l]
            Dict of encoding matrices for each CDR3 sequence length l.
        """
        position_vectors = dict()
        for cdr3_length in range(1,50): # maximum hashable sequence length = 50
            vector = positional_encoding(cdr3_length, self.m, self.p)
            position_vectors[cdr3_length] = vector
        return position_vectors

    def _pool(self, aa_vectors:np.ndarray, position_vectors:np.ndarray) -> np.ndarray:
        """
        Pool position-encoded AA hashes along y-axis.
        """
        return np.multiply(aa_vectors, position_vectors).sum(axis=0)


    @lru_cache(maxsize=None)
    def _hash_cdr3(self, cdr3: str) -> np.array:
        """
        Generate hash from AA sequence. Results are cached.

        Parameters
        ----------
        cdr3 : str
            The amino acid CDR3 sequence to hash.

        Returns
        -------
        np.array[m,]
            The resulting hash vector.
        """
        # trim CDR3 sequence
        r = -self.trim_right if self.trim_right else len(cdr3)
        cdr3 = cdr3[self.trim_left:r]
        # hash
        l = len(cdr3)
        aas = np.array([self.aa_vectors_[aa] for aa in cdr3])
        pos = self.position_vectors_[l]
        return self._pool(aas, pos)

    def _hash_collection(self, seqs: TcrCollection) -> np.array:
        """
        Generate hash for a group of CDR3 sequences, e.g. a TCR cluster or TCR repertoire.

        Parameters
        ----------
        cluster : TcrCollection, Iterable
            Cluster object containing sequences to hash.

        Returns
        -------
        np.array[64,]
            The resulting hash vector.
        """
        sequence_hashes = [self._hash_cdr3(cdr3) for cdr3 in seqs.cdr3s]
        return np.mean(sequence_hashes, axis=0)

    def transform(self, X: Union[TcrCollection, list, str], y=None) -> np.array:
        """
        Generate CDR3 hashes.

        Parameters
        ----------
        x : Union[TcrCollection, list, str]
            Objects to hash, this can be a single CDR3 sequence, but also a
            TcrCollection subclass or list thereof.

        Returns
        -------
        np.array[n,m]
            Array containing m-dimensional hashes for each of the n provided inputs.
        """
        check_is_fitted(self)
        if isinstance(X, (Repertoire, list, np.ndarray)):
            return np.array([self.transform(s) for s in X]).astype(np.float32)
        elif isinstance(X, Cluster):
            return self._hash_collection(X)
        elif isinstance(X, ClusteredRepertoire):
            return np.vstack([self.transform(s) for s in X]).astype(np.float32)
        else:
            return self._hash_cdr3(X)


class TCRDistEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        distance_matrix:np.array=TCRDIST_DM, 
        aa_dim:int=8,
        num_pos:int=16,
        n_trim:int=3,
        c_trim:int=2,
        cdr3_weight:int=3,
        organism:str='human',
        chain:str='B',
        full_tcr:bool=False
        ):
        """
        TCRDist-based vector embedding for amino acid sequences. Trims and
        gaps sequences to a fixed length and transforms the TCRDist-matrix
        into euclidean space, generating a unique embedding for distinct TCRs
        whose distances reflect the TCRDist-distances between the original sequences.

        Parameters
        ----------
        distance_matrix : np.ndarray[20,20]
            Twenty by twenty matrix containing AA distances. By default, the
            TCRDist matrix is used here.
        aa_dim : int
            Number of dimensions for each amino acid. The final vector will have a
            total length of aa_dim * num_pos dimensions.
        num_pos : int
            Fixed length to which a sequence is gapped after trimming.
        n_trim : int
            Number of amino acids trimmed from left side of sequence.
        c_trim : int
            Number of amino acids trimmed from right side of sequence.
        cdr3_weight : int
            Weighting factor for the CDR3 region relative to other CDRs.
            (Only when including the V gene)
        organism : str
            Organism from which the input sequences originate from.
        chain : str
            TCR chain from which the input sequences originate from.
        full_tcr : bool
            Boolean indicating whether the full TCR (CDR1+CDR2+CDR2.5+CDR3)
            should be used for creating the embedding. This parameter is
            primarily used to calculate the dimensionality of the final vector,
            which is necessary for indexing (see indexing.py classes).
        """
        self.distance_matrix = distance_matrix
        self.aa_dim = aa_dim
        self.num_pos = num_pos
        self.n_trim = n_trim
        self.c_trim = c_trim
        self.cdr3_weight = cdr3_weight
        self.organism = organism
        self.chain = chain
        
        self.full_tcr = full_tcr
        if self.full_tcr:
            self.m = self.aa_dim*self.num_pos + self.aa_dim*18
        else:
            self.m = self.aa_dim*self.num_pos 

    def __repr__(self):
        return f'TCRDistEncoder(aa_dim={self.aa_dim})'

    def _calc_mds_vecs(self, return_stress=False):
        '''
        Helper function to run MDS.
        '''
        mds = MDS(
            n_components=self.aa_dim,
            dissimilarity="precomputed",
            random_state=11,
            normalized_stress=False
            )
        vecs = mds.fit_transform(self.dm)
        if return_stress:
            return vecs, mds.stress_
        else:
            return vecs
    
    def _calc_tcrdist_aa_vectors(self, SQRT=True, verbose=False):
        '''
        Embed tcrdist distance matrix to Euclidean space.
        '''
        self.dm = np.zeros((21,21))
        self.dm[:20,:20] = self.distance_matrix
        self.dm[:20,20] = 4.
        self.dm[20,:20] = 4.
        if SQRT:
            self.dm = np.sqrt(self.dm) ## NOTE
        vecs, stress = self._calc_mds_vecs(return_stress=True)
        # print('vecs mean:', np.mean(vecs, axis=0)) #looks like they are already zeroed
        # vecs -= np.mean(vecs, axis=0) # I think this is unnecessary, but for sanity...
        if verbose:
            print(f'encoding tcrdist aa+gap matrix, dim= {self.aa_dim} stress= {stress}')
        return {aa:v for aa,v in zip(AALPHABET+GAPCHAR, vecs)}

    def _trim_and_gap_cdr3(self, cdr3):
        ''' 
        Convert a variable length cdr3 to a fixed-length sequence in a way
        that is consistent with tcrdist scoring, by trimming the ends and
        inserting gaps at a fixed position

        If the cdr3 is longer than num_pos + n_trim + c_trim, some residues will be dropped
        '''
        gappos = min(6, 3+(len(cdr3)-5)//2) - self.n_trim
        r = -self.c_trim if self.c_trim>0 else len(cdr3)
        seq = cdr3[self.n_trim:r]
        afterlen = min(self.num_pos-gappos, len(seq)-gappos)
        numgaps = max(0, self.num_pos-len(seq))
        fullseq = seq[:gappos] + GAPCHAR*numgaps + seq[-afterlen:]
        assert len(fullseq) == self.num_pos
        return fullseq

    # @lru_cache(maxsize=None)
    def _encode_sequence(self, seq):
        '''
        Convert a sequence to a vector by lining up the aa_vectors

        length of the vector will be dim * len(seq), where dim is the dimension of the
        embedding given by aa_vectors
        '''
        # self.calc_tcrdist_aa_vectors()
        dim = self.aa_vectors_['A'].shape[0]
        vec = np.zeros((len(seq)*dim,))
        for i,aa in enumerate(seq):
            vec[i*dim:(i+1)*dim] = self.aa_vectors_[aa]
        return vec

    # @lru_cache(maxsize=None)
    def _gapped_encode_cdr3(self, cdr3):
        '''
        Convert a cdr3 of variable length to a fixed-length vector
        by trimming/gapping and then lining up the aa_vectors

        length of the vector will be dim * num_pos, where dim is the dimension of the
        embedding given by aa_vectors
        '''
        return self._encode_sequence(self._trim_and_gap_cdr3(cdr3))
    
    # @lru_cache(maxsize=None)
    def _gapped_encode_tcr_chains(self, tcrs:pd.DataFrame) -> np.array:
        '''
        Convert a TCR (V gene + CDR3) of variable length to a fixed-length vector
        by trimming/gapping and then lining up the aa_vectors.

        Parameters
        ----------
        tcrs : pd.DataFrame
            DataFrame with V and CDR3 information in the named columns.
        '''
        self.tcrs = tcrs
        # !THE FOLLOWING V GENES CONTAIN '*' CHARACTER WHICH IS CAUSING ISSUES WITH THE ENCODING!
        # TRBV12-2*01 -----> FGH-NFFRS-*SIPDGSF
        # TRBV16*02 -------> KGH-S*FQN-ENVLPNSP
        to_remove = ['TRBV12-2*01','TRBV16*02']
        if self.tcrs[self.tcrs.v_call.isin(to_remove)].shape[0] > 0:
            print(f"WARNING: Removing TCRs with {to_remove}. This is a temporary measure to prevent KeyError caused by '*' character.\n")
            self.tcrs = self.tcrs[~self.tcrs.v_call.isin(to_remove)]

        vec_len = self.aa_dim * (self.num_pos_other_cdrs + self.num_pos)
        # Perhaps we should not print this message
        # print(
        #     f'gapped_encode_tcr_chains: aa_mds_dim={self.aa_dim}\n',
        #     f'num_pos_other_cdrs={self.num_pos_other_cdrs}',
        #     f'num_pos_cdr3={self.num_pos}', 
        #     f'vec_len={vec_len}'
        #     )

        vecs = []
        for v, cdr3 in zip(self.tcrs['v_call'], self.tcrs['junction_aa']):
            v_vec = self._encode_sequence(self.gene_cdr_strings[v])
            cdr3_vec = np.sqrt(self.cdr3_weight) * self._gapped_encode_cdr3(cdr3)
            vecs.append(np.concatenate([v_vec, cdr3_vec]))
        vecs = np.array(vecs)
        assert vecs.shape == (self.tcrs.shape[0], vec_len)
        return vecs

    # def _():


    def fit(self, X=None, y=None):
        self.aa_vectors_ = self._calc_tcrdist_aa_vectors()
        self.gene_cdr_strings = setup_gene_cdr_strings(self.organism, self.chain)
        self.num_pos_other_cdrs = len(next(iter(self.gene_cdr_strings.values())))
        assert all(len(x)==self.num_pos_other_cdrs for x in self.gene_cdr_strings.values())
        return self

    def transform(self, X: Union[TcrCollection, pd.DataFrame, list, str], y=None) -> np.array:
        """
        Generate CDR3 hashes.

        Parameters
        ----------
        x : Union[TcrCollection, list, str]
            Objects to hash, this can be a single CDR3 sequence, but also a
            TcrCollection subclass or list thereof.

        Returns
        -------
        np.array[n,m]
            Array containing m-dimensional hashes for each of the n provided inputs.
        """
        check_is_fitted(self)
        if isinstance(X, (list, np.ndarray)):
            return np.array([self.transform(s) for s in X]).astype(np.float32)
        elif isinstance(X, pd.DataFrame):
            if self.full_tcr:
                assert 'v_call' in X.columns, f"DataFrame does not include column named 'v_call'."
                assert 'junction_aa' in X.columns, f"DataFrame does not include column named 'junction_aa'."
                return self._gapped_encode_tcr_chains(X)
            else:
                assert 'junction_aa' in X.columns, f"DataFrame does not include column named 'junction_aa'."
                X = X.junction_aa.to_list()
                return np.array([self.transform(s) for s in X]).astype(np.float32)
        else:
            return self._gapped_encode_cdr3(X)