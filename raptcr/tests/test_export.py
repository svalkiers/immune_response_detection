"""
Test export result of vectorized TCRdist to sparse matrix
with dimensions matching input forground sequences. 

conda install pytest
conda install pytest-cov
pytest -v raptcr/tests/test_export.py
"""
import pandas as pd
import numpy as np
import os
import time
from scipy.sparse import csr_matrix

from raptcr.neighbors import NeighborEnrichment
from raptcr.export import index_neighbors_manual
from raptcr.export import range_search_to_csr_matrix

def test_export_to_csr_matrix():
    """
    integration test of (1) raptcr.export.index_neighbors_manual 
    with (2) raptcr.export.range_search_to_csr_matrix
    """
    start_time = time.perf_counter()
    foreground = pd.read_table('raptcr/datasets/1K_sequences.tsv')
    foreground['v_call'] = foreground['v_call'].apply(lambda x : f"{x}*01")
    foreground['j_call'] = foreground['j_call'].apply(lambda x : f"{x}*01")
    enricher = NeighborEnrichment(repertoire=foreground)
    enricher.fixed_radius_neighbors(radius=37) # Determine neighbors in foreground
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    assert(enricher.repertoire.shape[0] == foreground.shape[0])
    assert((foreground.junction_aa ==enricher.repertoire.junction_aa).all())
    lims, D, I  = index_neighbors_manual(query= enricher.repertoire, index=enricher.fg_index, r= enricher.r)
    csr_mat = range_search_to_csr_matrix(lims, D, I)
    assert(isinstance(csr_mat, csr_matrix))

