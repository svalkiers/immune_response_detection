import numpy as np 
import faiss
from scipy.sparse import csr_matrix

def index_neighbors_manual(query, r, index):
    '''
    For each query TCR sequence i, this finds neighbors on the index that are 
    within  a radius r of each query TCR. 
    
    Directly export the lim, D, I range_search output for manual 
    manipulation of neighbors.

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
    return lim, D, I 

def range_search_to_csr_matrix(lims,
    D, 
    I):
    """
    Convert output of faiss index.idx.range_search to a csr sparse matrix

    Parameters
    ----------
    lims
        output of index.idx.range_search, is range of positions in D and I corresponding 
        to the ith query sequence.
    D 
        neigbor distanace vector output of index.idx.range_search       
    I 
        neigbor index vector output of index.idx.range_search
    Returns
    -------
    csr_mat 
        scipy.sparse.csrmat of distance less than user specified max radius.
        zero distances will be encoded as -1 in the sparse csr_matrix 
    """
    dist_csr = list()
    for i in range(len(lims)-1):
        # <ix> is neighbor > r index position
        ix = I[lims[i]:lims[i+1]].tolist()
        # dx is neighbor > r distance
        dx = D[lims[i]:lims[i+1]].tolist()
        # convert distance to an integer
        dx_as_int = [round(x) for x in dx]
        # convert 0 dist to negative 1 for sparsity
        dx_as_int = [-1 if x == 0 else x for x in dx_as_int]
        dist_csr.append([(i,j,d) for j, d in zip(ix, dx_as_int)])
    # flatten list of tuple lists
    dist_csr = np.concatenate(dist_csr)
    row_indices = [row for row, col, val in dist_csr]
    col_indices = [col for row, col, val in dist_csr]
    values      = [val for row, col, val in dist_csr]
    csr_mat = csr_matrix((values, (row_indices, col_indices)))
    return csr_mat
