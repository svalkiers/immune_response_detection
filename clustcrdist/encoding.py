import numpy as np
import pandas as pd
import multiprocessing as mp

from typing import Union
from functools import lru_cache
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import MDS

from .repertoire import Repertoire
from .analysis import TcrCollection
from .constants.base import AALPHABET, GAPCHAR
from .constants.hashing import TCRDIST_DM
from .constants.preprocessing import (
    setup_gene_cdr_strings, detect_vgene_col, detect_cdr3_col
)

from sklearn.utils.validation import check_is_fitted

class TCRDistEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        distance_matrix:np.array=TCRDIST_DM, 
        aa_dim:int=8,
        vgene_col=None,
        cdr3_col=None,
        mds_eps:float=1e-05,
        num_pos:int=16,
        n_trim:int=3,
        c_trim:int=2,
        cdr3_weight:Union[int,float]=3,
        v_weight:Union[int,float]=1,
        organism:str='human',
        chain:str='B',
        full_tcr:bool=True,
        ncpus:int=1
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
        self.vgene_col = vgene_col
        self.cdr3_col = cdr3_col
        self.mds_eps = mds_eps
        self.num_pos = num_pos
        self.n_trim = n_trim
        self.c_trim = c_trim
        self.cdr3_weight = cdr3_weight
        self.v_weight = v_weight
        self.organism = organism
        self.ncpus = ncpus

        allowed_chains = [
            'a','b','g','d',
            'alpha','beta','gamma','delta',
            'ab','gd',
            'alphabeta','gammadelta']
        assert chain.lower() in allowed_chains, f'Invalid chain {chain}, please select alpha, beta, gamma, delta.'
        self.chain = chain
        self.full_tcr = full_tcr

    def __repr__(self):
        return f'TCRDistEncoder(aa_dim={self.aa_dim})'

    def _calc_mds_vecs(self, return_stress=False):
        '''
        Helper function to run MDS.
        '''
        mds = MDS(
            n_components=self.aa_dim,
            eps=self.mds_eps,
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

    def _gapped_encoder_v_cdr3(self, tcr : tuple) -> np.array:
        '''
        Calculate a numerical TCRdist-based encoding for a TCR (V call + junction amino acid)

        tcr, tuple
            Tuple containing V gene and CDR3 amino acid information.
            Example:  ("TRBV19*01", "CASSIGREAFF")
        '''
        v, cdr3 = tcr
        v_vec = np.sqrt(self.v_weight) * self._encode_sequence(self.gene_cdr_strings[v])
        cdr3_vec = np.sqrt(self.cdr3_weight) * self._gapped_encode_cdr3(cdr3)
        return np.concatenate([v_vec, cdr3_vec])
    
    # @lru_cache(maxsize=None)
    def _gapped_encode_tcr_chains(self, tcrs, vgene_col=None, cdr3_col=None) -> np.array:
        '''
        Convert a TCR (V gene + CDR3) of variable length to a fixed-length vector
        by trimming/gapping and then lining up the aa_vectors.

        Parameters
        ----------
        tcrs : pd.DataFrame
            DataFrame with V and CDR3 information in the named columns.
        '''
        # Prepare data structure
        self.tcrs = tcrs
        if vgene_col is None:
            vgene_col = detect_vgene_col(tcrs)
        if cdr3_col is None:
            cdr3_col = detect_cdr3_col(tcrs)
        if isinstance(tcrs, pd.DataFrame):
            tcrs = list(zip(tcrs[vgene_col], tcrs[cdr3_col]))
        else:
            pass
        # !THE FOLLOWING V GENES CONTAIN '*' CHARACTER WHICH IS CAUSING ISSUES WITH THE ENCODING!
        # TRBV12-2*01 -----> FGH-NFFRS-*SIPDGSF
        # TRBV16*02 -------> KGH-S*FQN-ENVLPNSP
        if self.organism == "human":
            to_remove = ['TRBV12-2*01','TRBV16*02']
            n_pre = len(tcrs)
            tcrs = [i for i in tcrs if i[0] not in to_remove]
            n_post = len(tcrs)
            if n_pre-n_post>0:
                print(f"WARNING: Removed TCRs with {to_remove}. This is a temporary measure to prevent KeyError caused by '*' character.\n")
        # Determine the length of the vector
        vec_len = self.aa_dim * (self.num_pos_other_cdrs + self.num_pos)
        # No parallel processing, use a single core
        if self.ncpus == 1:
            vecs = []
            for tcr in tcrs:
                v, cdr3 = tcr
                self._gapped_encoder_v_cdr3(tcr)
                v_vec = np.sqrt(self.v_weight) * self._encode_sequence(self.gene_cdr_strings[v])
                cdr3_vec = np.sqrt(self.cdr3_weight) * self._gapped_encode_cdr3(cdr3)
                vecs.append(np.concatenate([v_vec, cdr3_vec]))
            vecs = np.array(vecs)
        # Parallel processing, use n cores equal to self.ncpus
        else:
            print("mp")
            pool = mp.Pool(processes=self.ncpus)
            vecs = np.array(pool.map(self._gapped_encoder_v_cdr3, tcrs))
            pool.close()
            pool.join()
        # print(len(self.tcrs), vec_len)
        print('Shape of TCR vecs', vecs.shape)
        assert vecs.shape == (len(self.tcrs), vec_len)
        return vecs

    def encode_tcr(self, v, cdr3):
        v_vec = self._encode_sequence(self.gene_cdr_strings[v])
        cdr3_vec = np.sqrt(self.cdr3_weight) * self._gapped_encode_cdr3(cdr3)
        return np.concatenate([v_vec,cdr3_vec])
    
    def _encode_paired_chains(self, tcrs):
        avecs = self._gapped_encode_tcr_chains(tcrs[['va','cdr3a']],'va','cdr3a').astype(np.float32)
        bvecs = self._gapped_encode_tcr_chains(tcrs[['vb','cdr3b']],'vb','cdr3b').astype(np.float32)
        # Concatenate alpha & beta vectors
        abvecs = np.hstack([avecs, bvecs])
        assert abvecs.shape == (tcrs.shape[0], avecs.shape[1] + bvecs.shape[1])
        return abvecs

    def fit(self, X=None, y=None):
        self.aa_vectors_ = self._calc_tcrdist_aa_vectors()
        self.gene_cdr_strings = setup_gene_cdr_strings(self.organism, self.chain)
        self.num_pos_other_cdrs = len(next(iter(self.gene_cdr_strings.values())))
        assert all(len(x)==self.num_pos_other_cdrs for x in self.gene_cdr_strings.values())
        if self.full_tcr:
            self.m = self.aa_dim*self.num_pos + self.aa_dim*self.num_pos_other_cdrs
            if self.chain == 'AB':
                self.m *= 2
        else:
            self.m = self.aa_dim*self.num_pos 
        return self

    def transform(
            self, 
            X: Union[TcrCollection, pd.DataFrame, list, str], 
            split_ab=False,
            vgene_col = 'v_call',
            cdr3_col = 'junction_aa'
            ) -> np.array:
        """
        Generate TCRdist vectors.

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
                if self.chain == 'AB':
                    # if not in paired format
                    if not set(['va','vb','cdr3a','cdr3b']).issubset(X.columns):
                        assert 'locus' in X.columns, f"DataFrame must include column named 'locus'."
                        rep = Repertoire(X)
                        X = rep.airr_to_tcrdist_paired()
                    # split up alpha and beta vecs
                    if split_ab:
                        avecs = self._gapped_encode_tcr_chains(X, 'va', 'cdr3a').astype(np.float32)
                        bvecs = self._gapped_encode_tcr_chains(X, 'vb', 'cdr3b').astype(np.float32)
                        return avecs, bvecs
                    else:
                        return self._encode_paired_chains(X)
                else:            
                    # assert 'v_call' in X.columns, f"DataFrame is missing 'v_call' column."
                    # assert 'junction_aa' in X.columns, f"DataFrame is missing 'junction_aa' column."
                    return self._gapped_encode_tcr_chains(X,self.vgene_col,self.cdr3_col).astype(np.float32)
            else:
                assert 'junction_aa' in X.columns, f"DataFrame does not include column named 'junction_aa'."
                X = X.junction_aa.to_list()
                return np.array([self.transform(s) for s in X]).astype(np.float32)
        else:
            return self._gapped_encode_cdr3(X).astype(np.float32)
        
def join_ab_vecs(avecs,bvecs):
    return np.hstack([avecs, bvecs])