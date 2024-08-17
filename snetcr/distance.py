import numpy as np
import faiss

from timeit import default_timer as timer
from scipy.sparse import coo_matrix
from .encoding import TCRDistEncoder
from .indexing import convert_range_search_output

def compute_sparse_distance_matrix(tcrs, chain, organism, exact=True, r=96.5, m=8):
    '''
    Compute the sparse distance matrix for a set of TCRs.
    '''
    # Encode the TCRs
    t0 = timer()
    start = timer()
    encoder = TCRDistEncoder(aa_dim=m,organism=organism,chain=chain).fit()
    vecs = encoder.transform(tcrs).astype(np.float32)
    print(f'Encoding TCRs took {timer()-start:.2f}s')
    d = vecs.shape[1]

    if exact:
        # Flat index will ensure 'exact' search
        # However, this is still an approximation of the true TCRdist
        start = timer()
        index = faiss.IndexFlatL2(d)
        index.add(vecs)
        print(f'Building the index took {timer()-start:.2f}s')
    else:

        if len(vecs) > 100000:
            n = int(len(vecs)/10)
        else:
            n = min(len(vecs), 10000) 

        nlist = int(np.sqrt(len(vecs))) # number of clusters
        quantizer = faiss.IndexFlatL2(d) # the quantizer is a flat (L2) index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(vecs[np.random.choice(len(vecs), size=n, replace=False)])
        index.add(vecs)

    # Search index
    start = timer()
    lims, D, I = index.range_search(vecs, thresh=r)
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

# LEGACY
# def compute_sparse_distance_matrix(tcrs, chain, organism, exact=True, d=96.5, m=8):
#     '''
#     Compute the sparse distance matrix for a set of TCRs.
#     '''
#     # Encode the TCRs
#     t0 = timer()
#     start = timer()
#     encoder = TCRDistEncoder(aa_dim=m,organism=organism,chain=chain).fit()
#     vecs = encoder.transform(tcrs).astype(np.float32)
#     print(f'Encoding TCRs took {timer()-start:.2f}s')

#     if exact:
#         # Flat index will ensure 'exact' search
#         # However, this is still an approximation of the true TCRdist
#         start = timer()
#         index = FlatIndex(encoder=encoder)
#         index.add(tcrs, vecs)
#         print(f'Building the index took {timer()-start:.2f}s')
#     else:
#         # IVF index
#         start = timer()
#         k = round(len(vecs)/1000)
#         n = round(k/20)+2
#         if n > k:
#             n = k
#         index = IvfIndex(encoder=encoder, n_centroids=k, n_probe=n)
#         index.add(tcrs, vecs)
#         print(f'Building the index took {timer()-start:.2f}s')

#     # Search index
#     start = timer()
#     lims, D, I = index.idx.range_search(vecs, thresh=d)
#     print(f'Searching index within radius {d} took {timer()-start:.2f}s')

#     # Format the results.
#     start = timer() 
#     res = convert_range_search_output(lims, D, I)

#     # Determine the size of the matrix
#     max_index = len(tcrs)
#     base = [([i,i],0) for i in range(max_index)]
#     res += base
#     res = list(set((tuple(indices), distance-1) for indices, distance in res))

#     # Initialize lists to hold row indices, column indices, and data values
#     rows, cols, values = [], [], []

#     # Populate the lists with the given distances
#     for indices, distance in res:
#         i, j = indices
#         rows.append(i)
#         cols.append(j)
#         values.append(distance)
#         if i != j:  # Ensure the matrix is symmetric
#             rows.append(j)
#             cols.append(i)
#             values.append(distance)

#     # Convert the lists to a sparse matrix using coo_matrix
#     dm = coo_matrix((values, (rows, cols)), shape=(max_index, max_index), dtype=np.float64)
#     print(f'Building the sparse matrix took {timer()-start:.2f}s')
#     print(f'Total time to compute distance matrix: {timer()-t0:.2f}s')

#     return dm