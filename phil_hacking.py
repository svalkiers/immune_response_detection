######################################################################################88
import raptcr
import itertools as it
from raptcr.constants.hashing import BLOSUM_62
from raptcr.constants.base import AALPHABET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.manifold import MDS
from collections import Counter
from sys import exit
import sys
from os.path import exists
from glob import glob
from os import mkdir, system
import random

from phil_functions import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 110)
#pd.set_option('display.width', 167)



## testing new functions:

def read_yfv_tcrs(fname, min_count=2):
    ''' Read in the yellow fever datasets, Hutch copies can be found at
    rhino01:/loc/no-backup/pbradley/share/sebastiaan/yellow_fever/

    they were downloaded from
    https://github.com/mptouzel/pogorelyy_et_al_2018/tree/master/Yellow_fever
    '''
    print('reading:', fname)
    tcrs = pd.read_table(fname)
    tcrs.rename(columns={'Clone count':'count',
                         'All V hits':'v',
                         'All J hits':'j',
                         'N. Seq. CDR3':'cdr3nt',
                         'AA. Seq. CDR3':'cdr3aa',
                         }, inplace=True)
    mask = tcrs['count'] >= min_count
    print('filter by count:', mask.sum(), 'of', tcrs.shape[0])
    tcrs = tcrs[mask]
    tcrs['v'] = tcrs.v.str.split('*').str.get(0) + '*01'
    tcrs['j'] = tcrs.j.str.split('*').str.get(0) + '*01'

    tcrs['cdr3nt'] = tcrs.cdr3nt.str.lower()

    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, v_column, cdr3_column, organism, chain, j_column=j_column)
    print('num_tcrs:', tcrs.shape[0], fname)
    tcrs = tcrs['count v j cdr3nt cdr3aa'.split()]
    return tcrs

def read_reparsed_emerson_tcrs(fname):
    tcrs = pd.read_table(fname)
    tcrs = tcrs[tcrs.productive]
    tcrs.rename(columns={'v_gene':'v', 'j_gene':'j', 'cdr3':'cdr3aa',
                         'cdr3_nucseq':'cdr3nt'}, inplace=True)
    tcrs['count'] = 5 # dummy, these files don't have abundance information

    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, v_column, cdr3_column, organism, chain, j_column=j_column)
    print('num_tcrs:', tcrs.shape[0], fname)
    tcrs = tcrs['count v j cdr3nt cdr3aa'.split()]
    return tcrs

def get_original_18_ftags():
    fnames = glob('/home/pbradley/csdat/raptcr/slurm/run4/*_12.5_r0_fg*py')
    assert len(fnames) == 18
    return sorted(x.split('_')[-6] for x in fnames)

def get_original_18_fnames():
    ftags = get_original_18_ftags()
    dirname = '/home/pbradley/gitrepos/immune_response_detection/data/phil/britanova/'
    return [f'{dirname}{x}.gz' for x in ftags]


def calc_tcrdist_matrix(df1, df2, tcrdister):
    l1 = [(x.v, x.j, x.cdr3aa, x.cdr3nt) for x in df1.itertuples()]
    l2 = [(x.v, x.j, x.cdr3aa, x.cdr3nt) for x in df2.itertuples()]
    print('calc_tcrdist_matrix:', len(l1), len(l2))

    return np.array([tcrdister.single_chain_distance(a,b)
                     for a in l1 for b in l2]).reshape((len(l1), len(l2)))



def get_emerson_files_for_allele(allele):
    ''' allele is 'A*02:01' or something like that

    returns the filenames of all repertoires for subjects with that allele
    '''
    assert '*' in allele and ':' in allele and 'HLA' not in allele

    fname = '/home/pbradley/csdat/raptcr/emerson/newest_hla_info_df.tsv'
    df = pd.read_table(fname)

    dirname = '/home/pbradley/csdat/raptcr/emerson/'

    fnames = [f'{dirname}{row.hipid}_reparsed.tsv'
              for _, row in df.iterrows()
              if allele in row.tolist()]

    return fnames


def compute_leiden_clusters_from_vecs(vecs, num_nbrs=5, random_seed=20):
    ''' Convenience wrapper around leiden clustering
    '''
    import leidenalg
    import igraph as ig
    from scipy.spatial.distance import squareform, pdist
    #from sklearn.metrics import pairwise_distances

    num_groups = vecs.shape[0]
    g = ig.Graph(directed=True)
    g.add_vertices(num_groups)
    print('distances')
    distances = squareform(pdist(vecs))
    print('DONE distances')
    #distances = pairwise_distances(X2, metric='euclidean')
    for ii in range(num_groups):
        distances[ii,ii] = 10000
        inds = np.argsort(distances[ii,:])
        g.add_edges(list(zip([ii]*num_nbrs, inds[:num_nbrs])))
    # sources, targets = adjacency.nonzero()
    # weights = adjacency[sources, targets]
    # if isinstance(weights, np.matrix):
    #     weights = weights.A1
    # g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    # g.add_edges(list(zip(sources, targets)))
    partition_type = leidenalg.RBConfigurationVertexPartition
    print('leiden')
    part = leidenalg.find_partition(g, partition_type, seed=random_seed)
    print('DONE leiden')

    leiden_clusters = np.array(part.membership)
    return leiden_clusters

def compute_nbr_counts_flat(fg_vecs, bg_vecs, radius):
    ''' Convenience wrapper around neighbor counts calc
    '''
    assert fg_vecs.shape[1] == bg_vecs.shape[1]
    import faiss
    idx = faiss.IndexFlatL2(fg_vecs.shape[1])
    idx.add(bg_vecs)
    start = timer()
    lims,D,I = idx.range_search(fg_vecs, radius)
    print(f'flat index search {fg_vecs.shape[0]}x{bg_vecs.shape[0]} took'
          f' {timer()-start:.2f} seconds')
    nbr_counts = lims[1:] - lims[:-1]
    return nbr_counts


def read_yfv_expanded_clones():
    xclones = pd.read_table(
        '/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/yfv_expanded_clones.tsv')
    xclones.rename(columns={
        'CDR3.nucleotide.sequence':'cdr3nt',
        'bestVGene':'v',
        'bestJGene':'j',
        'CDR3.amino.acid.sequence':'cdr3aa',
        }, inplace=True)
    xclones['v'] = xclones.v+'*01'
    xclones['j'] = xclones.j+'*01'
    xclones['cdr3nt'] = xclones.cdr3nt.str.lower()
    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    xclones = filter_out_bad_genes_and_cdr3s(
        xclones, v_column, cdr3_column, organism, chain, j_column=j_column)
    print('num_xclones:', xclones.shape[0])
    return xclones


def read_yfv_alice_hits(min_count=2):
    fname = '/home/pbradley/csdat/raptcr/yfv/Pogorelyy_hits.tsv'
    tcrs = pd.read_table(fname)

    tcrs.rename(columns={
        'CDR3.nucleotide.sequence':'cdr3nt',
        'CDR3.amino.acid.sequence':'cdr3aa',
        'bestVGene':'v',
        'bestJGene':'j',
        'Read.count':'count',
        'donor':'sample_tag',
        }, inplace=True)
    tcrs['v'] = tcrs.v+'*01'
    tcrs['j'] = tcrs.j+'*01'
    tcrs['donor'] = tcrs.sample_tag.str.split('_').str.get(0)
    tcrs['timepoint'] = tcrs.sample_tag.str.split('_').str.get(1).astype(int)
    assert all(tcrs.sample_tag.str.split('_').str.get(2)=='F1')
    tcrs['cdr3nt'] = tcrs.cdr3nt.str.lower()
    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, v_column, cdr3_column, organism, chain, j_column=j_column)
    print('num ALICE hits:', tcrs.shape[0])
    #['donor', 'count', 'Read.proportion', 'cdr3nt', 'cdr3aa', 'v', 'j', 'D',
    # 'total_n', 'space', 'space_n', 'q', 'p_val']
    outcols = 'sample_tag donor timepoint count v j cdr3aa cdr3nt p_val'.split()
    return tcrs[tcrs['count']>=min_count][outcols]



def read_paired_data(fname):
    ''' updates nucseq cols; filters bad genes, cdr3s
    '''
    print('reading:', fname)
    tcrs = pd.read_table(fname)
    tcrs['cdr3a_nucseq'] = tcrs.cdr3a_nucseq.str.lower()
    tcrs['cdr3b_nucseq'] = tcrs.cdr3b_nucseq.str.lower()

    organism = 'human'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')
    return tcrs

######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88


if 0: # subset to tsang tcrs

    oldfiltfile = f'{DATADIR}phil/big_combo_tcrs_2024-02-02_filt.tsv'
    newfiltfile = f'{DATADIR}phil/big_combo_tcrs_2024-02-02_tsang.tsv'

    df = pd.read_table(oldfiltfile, low_memory=False)

    mask = df.cohort == 'p_tsang'
    print('subset to tsang tcrs:', mask.sum(), 'of', df.shape[0])

    df = df[mask].copy()

    df.to_csv(newfiltfile, sep='\t', index=False)
    print('made:', newfiltfile)
    exit()



if 0: # test background resampling functions

    fname = ('/home/pbradley/gitrepos/immune_response_detection/'
             'data/phil/britanova/A5-S18.txt.gz')

    tcrs = read_britanova_tcrs(fname)
    print('num_tcrs:', tcrs.shape[0])

    start = timer()
    junctions = parse_junctions_for_background_resampling(
        tcrs, 'human', 'B', 'v', 'j', 'cdr3aa', 'cdr3nt')
    print(f'parsing junctions took {timer()-start:.1f}s for {tcrs.shape[0]}')

    start = timer()
    bg_tcrs = resample_background_tcrs_v4('human', 'B', junctions)
    print(f'resampling took {timer()-start:.1f}s for {tcrs.shape[0]}')

    exit()



if 0: # clumping step 0b
    # rename tcrs file
    #
    # add this code to make it explicit what we must have done before, since
    # these files are identical

    '''rhino02 slurm$ md5sum  ~/csdat/big_covid/big_combo_tcrs_2023-03-07b.tsv  ~/gitrepos/immune_response_detection/data/phil/paired_sample_filt.tsv
f3945f0e901f7cc3fbfbbe3cf7e7801e  /home/pbradley/csdat/big_covid/big_combo_tcrs_2023-03-07b.tsv
f3945f0e901f7cc3fbfbbe3cf7e7801e  /home/pbradley/gitrepos/immune_response_detection/data/phil/paired_sample_filt.tsv'''

    ## this is for consistency with the big_covid pipeline, which expects a
    ## TCRs file and clumps file whose names "match"
    oldfile = DATADIR+'phil/big_combo_tcrs_2024-02-02_filt.tsv'
    newfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2024-02-02a.tsv'

    # oldfile = DATADIR+'phil/big_combo_tcrs_2024-01-18_filt.tsv'
    # newfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2024-01-18a.tsv'

    # oldfile = DATADIR+'phil/big_combo_tcrs_2023-12-17_filt.tsv'
    # newfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-12-17a.tsv'

    # oldfile = DATADIR+'phil/big_combo_tcrs_2023-11-22_filt.tsv'
    # newfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-11-22a.tsv'

    # oldfile = DATADIR+'phil/paired_sample_filt.tsv'
    # newfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-03-07b.tsv'

    assert exists(oldfile)
    assert not exists(newfile)

    cmd = f'cp {oldfile} {newfile}'
    print(cmd)
    system(cmd)

    exit()


if 1: # clumping step 3
    # try neighborhood clustering of the clumping hits

    import igraph as ig
    import faiss

    organism = 'human'
    aa_mds_dim = 8

    #leiden_radii = [24]
    leiden_radii = [24,48,72,96]

    min_group_size = 10

    max_impurity = 0.2

    evalue_threshold= 1.

    classic2new_colmap = {
        'clone_index':'tcr_index',
        'pvalue_adj':'evalue',
        'nbr_radius':'radius',
    }


    if 2:
        # max_impurity=0.2, CLASSIC=False, evalue_threshold = 1.
        fname = DATADIR+'phil/zenaim_filt_clumping_pvals_run105_run106_evt_100.tsv'
        renamed_outfile = '/home/pbradley/csdat/koelle/zenaim/clumps.tsv' ; min_group_size = 3
        # fname = DATADIR+'phil/big_combo_tcrs_2024-02-02_tsang_clumping_pvals_run103_run104_evt_100.tsv'
        # renamed_outfile = ('/home/pbradley/csdat/big_covid/'
        #                    'big_combo_tcrs_2024-02-02a_tsang_gp4.tsv')
        # fname = DATADIR+'phil/big_combo_tcrs_2024-02-02_filt_clumping_pvals_run59_run61_evt_100.tsv'
        # renamed_outfile = ('/home/pbradley/csdat/big_covid/'
        #                    'big_combo_tcrs_2024-02-02a_gp4.tsv')
        # fname = DATADIR+'phil/big_combo_tcrs_2024-01-18_filt_clumping_pvals_run55_run56_evt_100.tsv'
        # renamed_outfile = ('/home/pbradley/csdat/big_covid/'
        #                    'big_combo_tcrs_2024-01-18a_gp4.tsv')
        # fname = DATADIR+'phil/big_combo_tcrs_2023-12-17_filt_clumping_pvals_run53_run54_evt_100.tsv'
        # renamed_outfile = ('/home/pbradley/csdat/big_covid/'
        #                    'big_combo_tcrs_2023-12-17a_gp4.tsv')
        # fname = DATADIR+'phil/big_combo_tcrs_2023-11-22_filt_clumping_pvals_run48_run49_evt_100.tsv'
        # fname = DATADIR+'phil/paired_sample_filt_clumping_pvals_run20_run21_evt_100.tsv'
        CLASSIC = False
    else:
        fname = ('/home/pbradley/csdat/big_covid/'
                 'big_combo_tcrs_2023-03-07_clumping_results_vjpair_nr1M_P100.tsv')
        CLASSIC = True
        # legacy: will fail now since renamed_outfile undefined

    lrtag = '_'.join(map(str,leiden_radii))
    outfile = (f'{fname[:-4]}_r{lrtag}_evt_{evalue_threshold:.1f}_'
               f'mi_{max_impurity:.3f}_nbrhood.tsv')



    # oops this one got overwritten!!!! need to re-run if we want it again
    # # 22a max_impurity=0.2, CLASSIC=False, evalue_threshold = 1.
    # renamed_outfile = ('/home/pbradley/csdat/big_covid/'
    #                    'big_combo_tcrs_2023-11-22a_gp4.tsv') # matches block above


    # 7e: max_impurity=0.2 ; classic clumping results
    # renamed_outfile = ('/home/pbradley/csdat/big_covid/'
    #                    'big_combo_tcrs_2023-03-07e_gp4.tsv')

    # 7d: max_impurity=0.2
    # renamed_outfile = ('/home/pbradley/csdat/big_covid/'
    #                    'big_combo_tcrs_2023-03-07d_gp4.tsv')

    # 7c had no impurity threshold
    # renamed_outfile = ('/home/pbradley/csdat/big_covid/'
    #                    'big_combo_tcrs_2023-03-07c_gp4.tsv')

    results = pd.read_table(fname)

    if CLASSIC:
        results.rename(columns=classic2new_colmap, inplace=True)

    results['impurity'] = results.expected_num_nbrs / results.num_nbrs
    print('start num tcrs:', len(results.tcr_index.unique()))
    results = results[results.evalue <= evalue_threshold]
    print('pval_thresh num tcrs:', len(results.tcr_index.unique()))
    results = results[results.impurity <= max_impurity]
    print('impurity_thresh num tcrs:', len(results.tcr_index.unique()))


    tcrs = results.sort_values(['evalue','radius']).drop_duplicates('tcr_index')
    tcr_index2tcrs_index = {x:i for i,x in enumerate(tcrs.tcr_index)}
    results['tcrs_index'] = results.tcr_index.map(tcr_index2tcrs_index)

    tcrs.reset_index(drop=True, inplace=True)

    avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    vecs = np.hstack([avecs, bvecs])

    # create the graph
    g = ig.Graph(directed=False) # undirected graph
    num_tcrs = vecs.shape[0]
    g.add_vertices(num_tcrs)

    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)

    for radius in leiden_radii:
        # which tcrs are significant at this radius?
        inds = np.array(sorted(set(results[results.radius==radius].tcrs_index)))
        #inds = np.nonzero(np.array(tcrs.radius==radius))[0]

        start = timer()
        print(f'radius {radius} num {inds.shape[0]} range search')
        lims, D, I = idx.range_search(vecs[inds], radius + .5)
        print(f'range_search took {timer()-start:.2f}')
        print('total nbrs:', radius, I.shape, flush=True)
        nnbrs = lims[1:]-lims[:-1]
        I0 = np.repeat(inds, nnbrs.astype(int))
        start = timer()
        assert I0.shape == I.shape
        g.add_edges(list(zip(I0,I)))
        # for ii, (start, stop) in enumerate(zip(lims[:-1], lims[1:])):
        #     g.add_edges(list(zip(it.repeat(inds[ii]), I[start:stop])))
        print(f'add nbrs took {timer()-start:.2f}', flush=True)


    # iteratively find the highest-degee vertex and remove it and its nbrs
    # these are the groups
    NOGROUP = -1
    group_numbers = np.full((num_tcrs,), NOGROUP, dtype=int)
    vertex2tcr = list(range(num_tcrs)) # map from graph vertices to tcr numbers
    centers = []
    while True:
        assert len(vertex2tcr) == g.vcount()
        degreelist = g.degree()
        center = np.argmax(degreelist)
        nbrs = sorted(set(g.neighbors(center)))
        assert center in nbrs
        print(g.vcount(), center, degreelist[center], g.maxdegree(), len(nbrs))
        tcr_nbrs = [vertex2tcr[x] for x in nbrs]
        tcr_center = vertex2tcr[center]
        assert np.all(group_numbers[tcr_nbrs]==NOGROUP)
        group_numbers[tcr_nbrs] = len(centers)
        centers.append(tcr_center)
        g.delete_vertices(nbrs)
        for v in sorted(nbrs, reverse=True):
            vertex2tcr.pop(v)
        if g.vcount()==0:
            break
        # some groups are smaller than this, maybe because there can be multiple edges
        # between nodes if they are closer than smaller distance thresholds...
        # right, for each threshold we add edges between all pairs of tcrs
        # that are significant at that threshold and closer than (<=) that threshold
        #
        if g.maxdegree()+1 < min_group_size:
            break

    mask = group_numbers==NOGROUP
    if mask.sum():
        next_group = max(group_numbers)+1
        group_numbers[mask] = np.arange(next_group, next_group+mask.sum())

    assert group_numbers.shape[0] == tcrs.shape[0]

    tcr_index2clumping_group = dict(zip(tcrs.tcr_index, group_numbers))

    results['clumping_group'] = results.tcr_index.map(tcr_index2clumping_group)

    results.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)


    ## now create a renamed version for compatibility with the big_covid.py pipeline:
    #
    #head -1 big_combo_tcrs_2023-03-07_clumping_results_vjpair_nr1M_P100.tsv
    #clump_type	clone_index	nbr_radius	pvalue_adj	num_nbrs	expected_num_nbrs	raw_count	va	ja	cdr3a	vb	jb	cdr3b	clonotype_fdr_value	clumping_group

    #head -1 ~/gitrepos/immune_response_detection/data/phil/paired_sample_filt_clumping_pvals_run20_run21_evt_100.tsv
    #evalue	tcr_index	radius	num_nbrs	expected_num_nbrs	bg_nbrs	va	ja	cdr3a	cdr3a_nucseq	vb	jb	cdr3b	cdr3b_nucseq	cohort
    if not CLASSIC:
        results['clump_type'] = 'global'
        results.drop(columns=['cdr3a_nucseq','cdr3b_nucseq'], inplace=True)

    results.rename(columns={y:x for x,y in classic2new_colmap.items()}, inplace=True)
    results.to_csv(renamed_outfile, sep='\t', index=False)
    print('made:', renamed_outfile)



    exit()


if 0: # try single-linkage clustering of the clumping hits
    import igraph as ig
    import faiss

    organism = 'human'
    aa_mds_dim = 8
    evalue_threshold= 1.

    #leiden_radii = [24]
    leiden_radii = [24,48,72,96]

    fname = DATADIR+'phil/big_combo_tcrs_2024-02-02_tsang_clumping_pvals_run103_run104_evt_100.tsv'
    renamed_outfile = ('/home/pbradley/csdat/big_covid/'
                       'big_combo_tcrs_2024-02-02b_tsang_gp4.tsv')

    #fname = DATADIR+'phil/paired_sample_filt_clumping_pvals_run20_run21_evt_100.tsv'
    # use 2023-03-07b !!!
    #renamed_outfile = ('/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-03-07b_'
    #                   'clumping_results_vjpair_nr1M_P100.tsv')
    lrtag = '_'.join(map(str,leiden_radii))
    outfile = f'{fname[:-4]}_r{lrtag}_single.tsv'

    results = pd.read_table(fname)
    results = results[results.evalue <= evalue_threshold]

    tcrs = results.sort_values(['evalue','radius']).drop_duplicates('tcr_index')
    tcr_index2tcrs_index = {x:i for i,x in enumerate(tcrs.tcr_index)}
    results['tcrs_index'] = results.tcr_index.map(tcr_index2tcrs_index)

    tcrs.reset_index(drop=True, inplace=True)

    avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    vecs = np.hstack([avecs, bvecs])

    # create the graph
    g = ig.Graph(directed=False) # undirected graph
    g.add_vertices(vecs.shape[0])


    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)

    for radius in leiden_radii:
        # which tcrs are significant at this radius?
        inds = np.array(sorted(set(results[results.radius==radius].tcrs_index)))
        #inds = np.nonzero(np.array(tcrs.radius==radius))[0]

        start = timer()
        print(f'radius {radius} num {inds.shape[0]} range search')
        lims, D, I = idx.range_search(vecs[inds], radius + .5)
        print(f'range_search took {timer()-start:.2f}')
        print('total nbrs:', radius, I.shape, flush=True)
        nnbrs = lims[1:]-lims[:-1]
        I0 = np.repeat(inds, nnbrs.astype(int))
        start = timer()
        assert I0.shape == I.shape
        g.add_edges(list(zip(I0,I)))
        # for ii, (start, stop) in enumerate(zip(lims[:-1], lims[1:])):
        #     g.add_edges(list(zip(it.repeat(inds[ii]), I[start:stop])))
        print(f'add nbrs took {timer()-start:.2f}', flush=True)

    comps = g.connected_components() # mode=weak or strong is ignored for undirect graph
    assert len(comps.membership) == tcrs.shape[0]

    # rename by size order
    sizes = Counter(comps.membership)
    old2new = {g:ii for ii,(g,_) in enumerate(sizes.most_common())}

    tcr_index2clumping_group = dict(zip(
        tcrs.tcr_index, [old2new[x] for x in comps.membership]))

    results['clumping_group'] = results.tcr_index.map(tcr_index2clumping_group)

    results.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)


    ## now create a renamed version for compatibility with the big_covid.py pipeline:
    #
    #head -1 big_combo_tcrs_2023-03-07_clumping_results_vjpair_nr1M_P100.tsv
    #clump_type	clone_index	nbr_radius	pvalue_adj	num_nbrs	expected_num_nbrs	raw_count	va	ja	cdr3a	vb	jb	cdr3b	clonotype_fdr_value	clumping_group

    #head -1 ~/gitrepos/immune_response_detection/data/phil/paired_sample_filt_clumping_pvals_run20_run21_evt_100.tsv
    #evalue	tcr_index	radius	num_nbrs	expected_num_nbrs	bg_nbrs	va	ja	cdr3a	cdr3a_nucseq	vb	jb	cdr3b	cdr3b_nucseq	cohort
    results['clump_type'] = 'global'
    results.drop(columns=['cdr3a_nucseq','cdr3b_nucseq'], inplace=True)
    results.rename(columns={
        'evalue':'pvalue_adj',
        'tcr_index':'clone_index',
        'radius':'nbr_radius'}, inplace=True)
    results.to_csv(renamed_outfile, sep='\t', index=False)
    print('made:', renamed_outfile)



    exit()


if 0:
    fname = DATADIR+'phil/paired_sample_filt_clumping_pvals_run20_run21_evt_100_leidens_uniqtcrs.r24.tsv'

    fname = DATADIR+'phil/paired_sample_filt_clumping_pvals_run20_run21_evt_100_r24_48_72_96_leidens_uniqtcrs.tsv'

    tcrs = pd.read_table(fname)

    for c in range(10):
        mask = tcrs.leiden==c
        if mask.sum()<20:
            break
        print('='*80)
        print(mask.sum(), c)
        print(tcrs[mask]['va cdr3a vb cdr3b cohort'.split()])
        print(tcrs[mask].cohort.value_counts().head())



    exit()


if 0: # try leiden clustering of the clumping tcrs
    import leidenalg
    import igraph as ig
    organism = 'human'
    aa_mds_dim = 8

    #leiden_radii = [24]
    leiden_radii = [24,48,72,96]

    threshold= 1.
    fname = DATADIR+'phil/paired_sample_filt_clumping_pvals_run20_run21_evt_100.tsv'
    lrtag = '_'.join(map(str,leiden_radii))
    outfile = f'{fname[:-4]}_r{lrtag}_leidens.tsv'
    df = pd.read_table(fname)
    df = df[df.evalue <= threshold]

    tcrs = df.sort_values(['evalue','radius']).drop_duplicates('tcr_index')

    tcrs.reset_index(drop=True, inplace=True)

    avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    vecs = np.hstack([avecs, bvecs])

    # create the graph
    g = ig.Graph(directed=True)
    g.add_vertices(vecs.shape[0])


    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)

    for radius in leiden_radii:
        inds = np.nonzero(np.array(tcrs.radius==radius))[0]

        start = timer()
        print(f'radius {radius} num {inds.shape[0]} range search')
        lims, D, I = idx.range_search(vecs[inds], radius + .5)
        print(f'range_search took {timer()-start:.2f}')
        print('total nbrs:', radius, I.shape, flush=True)
        nnbrs = lims[1:]-lims[:-1]
        I0 = np.repeat(inds, nnbrs.astype(int))
        stime = timer()
        assert I0.shape == I.shape
        g.add_edges(list(zip(I0,I)))
        # for ii, (start, stop) in enumerate(zip(lims[:-1], lims[1:])):
        #     g.add_edges(list(zip(it.repeat(inds[ii]), I[start:stop])))
        print(f'add nbrs took {timer()-stime:.2f}', flush=True)

    partition_type = leidenalg.RBConfigurationVertexPartition
    print('leiden')
    start = timer()
    part = leidenalg.find_partition(g, partition_type, seed=11)#random_seed)
    print(f'DONE leiden {timer()-start:.2f}')

    leiden_clusters = np.array(part.membership)

    tcrs['leiden'] = leiden_clusters

    df = df.join(tcrs.set_index('tcr_index')['leiden'], on='tcr_index')
    df.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

    tcrs.to_csv(outfile[:-4]+'_uniqtcrs.tsv', sep='\t', index=False)

    exit()


if 0: # find cohorts with highest density of clumping hits
    threshold= 1.
    fname = DATADIR+'phil/paired_sample_filt_clumping_pvals_run20_run21_evt_100.tsv'

    df = pd.read_table(fname)
    df = df[df.evalue <= threshold]

    print('reading tcrs')
    tcrs = pd.read_table(f'{DATADIR}phil/paired_sample_filt.tsv')
    cohort_counts = tcrs.cohort.value_counts()
    print('cohort_counts:', cohort_counts.head())
    #df = df.join(tcrs, on='tcr_index')

    cohort_hits = df.drop_duplicates('tcr_index').cohort.value_counts()

    l = []
    for cohort, num in cohort_hits.items():
        rate = num / cohort_counts[cohort]
        l.append((rate, cohort))

    for (rate, cohort) in sorted(l, reverse=True):
        print(f'{100*rate:7.3f}% {cohort_hits[cohort]:5d} {cohort_counts[cohort]:6d}',
              cohort)




    exit()

if 0: # clumping step 2
    # read the results of the paired tcr clumping calcs
    from scipy.stats import poisson
    jobsize = 10000


    fg_runtag = 'run105' # range_search with fg_vecs
    bg_runtag = 'run106' # single-chain matching to bg a+b vecs
    fg_filename = f'{DATADIR}phil/zenaim_filt.tsv'
    num_fg_tcrs = 18422 # in fg_filename
    njobs = 1 ; jobsize = 30000
    # fg_runtag = 'run103' # range_search with fg_vecs
    # bg_runtag = 'run104' # single-chain matching to bg a+b vecs
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2024-02-02_tsang.tsv'
    # num_fg_tcrs = 84369 # in fg_filename
    # njobs = 9
    # fg_runtag = 'run59' # range_search with fg_vecs
    # bg_runtag = 'run61' # single-chain matching to bg a+b vecs
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2024-02-02_filt.tsv'
    # num_fg_tcrs = 3971185 # in fg_filename
    # njobs = 398
    # fg_runtag = 'run55' # range_search with fg_vecs
    # bg_runtag = 'run56' # single-chain matching to bg a+b vecs
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2024-01-18_filt.tsv'
    # num_fg_tcrs = 3882371 # in fg_filename
    # njobs = 389
    # fg_runtag = 'run53' # range_search with fg_vecs
    # bg_runtag = 'run54' # single-chain matching to bg a+b vecs
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2023-12-17_filt.tsv'
    # num_fg_tcrs = 3926238 # in fg_filename
    # njobs = 393
    # fg_runtag = 'run48' # range_search with fg_vecs
    # bg_runtag = 'run49' # single-chain matching to bg a+b vecs
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2023-11-22_filt.tsv'
    # num_fg_tcrs = 3889692  # in fg_filename
    # njobs = 389
    # fg_runtag = 'run20' # range_search with fg_vecs
    # bg_runtag = 'run21' # single-chain matching to bg a+b vecs
    # fg_filename = f'{DATADIR}phil/paired_sample_filt.tsv'
    # num_fg_tcrs = 3097957  # in fg_filename
    # njobs = 310
    #jobsize = 10000
    num_bg_tcrs = 250000
    maxdist = 96
    radii = [24, 48, 72, 96]
    ab_veclen = 568
    min_nbrs = 2 # tcr_clumping uses 1?
    pseudocount = 0.25
    evalue_threshold = 100

    assert njobs == (num_fg_tcrs-1)//jobsize + 1 # sanity, or could just calc here!

    print('reading:', fg_filename)
    tcrs = pd.read_table(fg_filename)
    assert tcrs.shape[0] == num_fg_tcrs
    if 'cohort' not in tcrs.columns:
        print('make fake cohort data')
        tcrs['cohort'] = 'cohort_1'

    outfile = (f'{fg_filename[:-4]}_clumping_pvals_{fg_runtag}_{bg_runtag}_'
               f'evt_{evalue_threshold}.tsv')

    def load_bg_counts(jobno, runtag=bg_runtag):
        ''' Returns array (rcounts) of size (ntcrs, len(radii))
        with the summed counts up to and including each radius value
        '''
        fname = (f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
                 f'{runtag}_{jobno}_{jobno*jobsize}_{(jobno+1)*jobsize}_'
                 f'{num_bg_tcrs}_ab_counts.npy')
        counts = np.load(fname)
        ntcrs = counts.shape[0]
        assert counts.shape[1] == maxdist+1
        assert jobno == njobs-1 or ntcrs == jobsize
        rcounts = np.cumsum(counts, axis=1)[:, radii]
        return rcounts

    def load_fg_counts(jobno, runtag=fg_runtag):
        actual_jobsize = jobsize if jobno<njobs-1 else num_fg_tcrs%jobsize
        prefix = (f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
                  f'{runtag}_{jobno}_{maxdist+0.5:.1f}_{actual_jobsize}_{num_fg_tcrs}_'
                  f'{ab_veclen}_')
        #fname = run20_23_96.5_10000_3097957_568_D.npy
        lims = np.load(prefix+'lims.npy')
        D = np.load(prefix+'D.npy')
        #I = np.load(prefix+'I.npy')

        ntcrs = lims.shape[0]-1
        assert ntcrs==jobsize or jobno==njobs-1

        rcounts = np.zeros((ntcrs, len(radii)), dtype=int)
        for ii, r in enumerate(radii):
            rcounts[:,ii] = np.add.reduceat((D<=r+.5), lims[:-1].astype(int))
        return rcounts


    dfl = []
    for jobno in range(njobs):
        startnum = len(dfl)

        fg_counts = load_fg_counts(jobno) - 1 # exclude self nbrs
        bg_counts = load_bg_counts(jobno)

        rates = (np.maximum(bg_counts, pseudocount) *
                 (num_fg_tcrs/(num_bg_tcrs*num_bg_tcrs)))

        for ii, (counts,rates) in enumerate(zip(fg_counts, rates)):
            for jj, (count,rate) in enumerate(zip(counts,rates)):
                if count >= min_nbrs:
                    pval = poisson.sf(count-1, rate)
                    pval *= len(radii)*num_fg_tcrs
                    if pval <= evalue_threshold:
                        dfl.append(dict(
                            evalue=pval,
                            tcr_index=ii+jobno*jobsize,
                            radius=radii[jj],
                            num_nbrs=count,
                            expected_num_nbrs=rate,
                            bg_nbrs=rate*num_bg_tcrs*num_bg_tcrs/num_fg_tcrs,
                        ))
        inds = sorted(set(x['tcr_index'] for x in dfl[startnum:]))
        ccounts = Counter(tcrs.cohort.iloc[inds]).most_common(3)
        ccountstring = ' '.join(f'{x}:{i}' for x,i in ccounts)
        print(f'{jobno:3d} {len(dfl)-startnum:4d} {len(dfl)} {ccountstring}')
    results = pd.DataFrame(dfl).join(tcrs, on='tcr_index')
    results.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)


    exit()

if 0: # experiment with faiss distances, for calculating the pairwise bg distns
    # python timing, around 2.7 seconds:
    #
    # allocate memory 10000 50000
    # start pairwise_L2sqr nq= 10000 nb= 50000 dim= 272
    # finished: 2.62
    #
    # vs in C++ about the same (these are in milliseconds; see phil_hacking.cpp)
    #
    # rhino01 immune_response_detection$ ./a.out
    # pairwise dists searching... 10000 x 50000
    # DONE dists searching... 2569.433838
    # rhino01 immune_response_detection$ ./a.out
    # pairwise dists searching... 10000 x 50000
    # DONE dists searching... 2558.998047

    import faiss
    from scipy.spatial.distance import cdist
    from sklearn.metrics import pairwise_distances

    nq = 10000
    nb = 50000
    dim = 272

    xq = np.random.rand(nq, dim).astype(np.float32)
    xb = np.random.rand(nb, dim).astype(np.float32)
    print('allocate memory', nq, nb)
    result = np.zeros((nq, nb), dtype=np.float32)

    for r in range(5):
        print(f'start pairwise_L2sqr nq= {nq} nb= {nb} dim= {dim}')
        start = timer()
        faiss.pairwise_L2sqr(dim, nq, faiss.swig_ptr(xq), nb, faiss.swig_ptr(xb),
                             faiss.swig_ptr(result))
        print(f'finished1: {timer()-start:.2f}', flush=True)

        print(f'start pairwise_L2sqr nq= {nq} nb= {nb} dim= {dim}')
        start = timer()
        D = faiss.pairwise_distances(xq, xb)
        print(f'finished1: {timer()-start:.2f}', flush=True)

        print(f'start pairwise_distances nq= {nq} nb= {nb} dim= {dim}')
        start = timer()
        D = pairwise_distances(xq, xb)
        print(f'finished2: {timer()-start:.2f}', flush=True)

        print(f'start cdist nq= {nq} nb= {nb//10} dim= {dim}')
        start = timer()
        D = cdist(xq, xb[:nb//10])
        print(f'finished3: {timer()-start:.2f}', flush=True)

        print(f'start cdist nq= {nq} nb= {nb} dim= {dim}')
        start = timer()
        D = cdist(xq, xb)
        print(f'finished4: {timer()-start:.2f}', flush=True)

    exit()

    results = '''
In [105]: %run phil_hacking.py
allocate memory 10000 50000
start pairwise_L2sqr nq= 10000 nb= 50000 dim= 272
finished: 2.60
start pairwise_distances nq= 10000 nb= 50000 dim= 272
finished: 7.80
start cdist nq= 10000 nb= 5000 dim= 272
finished: 5.63

In [106]: %run phil_hacking.py
allocate memory 10000 50000
start pairwise_L2sqr nq= 10000 nb= 50000 dim= 272
finished: 3.19
start pairwise_distances nq= 10000 nb= 50000 dim= 272
finished: 8.14
start cdist nq= 10000 nb= 5000 dim= 272
finished: 5.81

In [107]: %run phil_hacking.py
allocate memory 10000 50000
start pairwise_L2sqr nq= 10000 nb= 50000 dim= 272
finished: 2.97
start pairwise_distances nq= 10000 nb= 50000 dim= 272
finished: 8.22
start cdist nq= 10000 nb= 5000 dim= 272
finished: 5.70

    older:

In [96]: %run phil_hacking.py
allocate memory 10000 50000
start pairwise_L2sqr nq= 10000 nb= 50000 dim= 272
finished: 3.27
start cdist nq= 10000 nb= 50000 dim= 272
finished: 399.98

In [97]: %run phil_hacking.py
allocate memory 10000 50000
start pairwise_L2sqr nq= 10000 nb= 50000 dim= 272
finished: 2.85
start cdist nq= 10000 nb= 50000 dim= 272
finished: 192.54


    '''


if 0: # clumping step 1
    # setup for some background distn searches
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    maxdist = 96
    num_bg_tcrs = 250000
    aa_mds_dim = 8
    job_size = 10000
    runtag = 'run106' ; job_size  = 30000
    fg_filename = f'{DATADIR}phil/zenaim_filt.tsv'
    # runtag = 'run104'
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2024-02-02_tsang.tsv'
    # runtag = 'run61'
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2024-02-02_filt.tsv'
    # runtag = 'run56'
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2024-01-18_filt.tsv'
    # runtag = 'run54'
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2023-12-17_filt.tsv'
    # runtag = 'run49'
    # fg_filename = f'{DATADIR}phil/big_combo_tcrs_2023-11-22_filt.tsv'
    # runtag = 'run21'
    # fg_filename = f'{DATADIR}phil/paired_sample_filt.tsv'
    xargs = ' --sleeptime 30 '

    bg_filename = f'{DATADIR}phil/paired_bg_tcrs_250000.tsv'

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    assert exists(bg_filename) and exists(fg_filename)
    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    print('reading:', fg_filename)
    tcrs = pd.read_table(fg_filename)
    num_tcrs = tcrs.shape[0]

    num_jobs = (num_tcrs-1)//job_size+1


    for job in range(num_jobs):
        start = job*job_size
        stop = (job+1)*job_size

        outfile = f'{rundir}{runtag}_{job}_{start}_{stop}_{num_bg_tcrs}_ab_counts.npy'
        cmd = (f'{PY} {EXE} {xargs} --mode paired_backgrounds '
               f' --fg_filename {fg_filename} --bg_filename {bg_filename} '
               f' --maxdist {maxdist} --num_bg_tcrs {num_bg_tcrs} '
               f' --start_index {start} --stop_index {stop} '
               f' --aa_mds_dim {aa_mds_dim} '
               f' --outfile {outfile} '
               f' > {outfile[:-4]}.log 2> {outfile[:-4]}.err')
        out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)
    exit()


if 0: # metaconga step 6
    # filter the tcrs from the round4 or round6... merged file
    # so we can get TCR KNNs for hotspot analysis
    bigfile = '/home/pbradley/csdat/big_covid/round8_merged_common_genes_obs.tsv'
    filtfile = f'{DATADIR}phil/round8_merged_filt.tsv'
    # bigfile = '/home/pbradley/csdat/big_covid/round6_merged_common_genes_obs.tsv'
    # filtfile = f'{DATADIR}phil/round6_merged_filt.tsv'
    # bigfile = '/home/pbradley/csdat/big_covid/round4_merged_common_genes_obs.tsv'
    # filtfile = f'{DATADIR}phil/round4_merged_filt.tsv'

    print('reading:', bigfile)
    tcrs = pd.read_table(bigfile, low_memory=False)
    old_num = tcrs.shape[0]
    print('read:', old_num, bigfile)

    tcrs['cell_index'] = tcrs.index
    tcrs.rename(columns={tcrs.columns[0]:'barcode'}, inplace=True)

    tcrs.sort_values(['cohort','tcr_string','clone_size'], ascending=False,
                     inplace=True)
    tcrs['clone_count'] = tcrs.groupby(['cohort','tcr_string']).cumcount()
    mask = tcrs.clone_count==0

    print('filter to clones:', old_num, mask.sum())
    tcrs = tcrs[mask]

    tcrs['cdr3a_nucseq'] = tcrs.cdr3a_nucseq.str.lower()
    tcrs['cdr3b_nucseq'] = tcrs.cdr3b_nucseq.str.lower()
    organism = 'human'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')
    tcrs.to_csv(filtfile, sep='\t', index=False)
    print('made:', old_num, mask.sum(), tcrs.shape[0], filtfile)

    exit()

if 0: # setup for some knn searches from vector inputs (here metaconga GEX pcs)
    # multi-runtag version
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    job_size = 10000
    xargs = ' --sleeptime 30 '
    mode = 'vector_knns'
    num_nbrs = 501
    bcdir = '/home/pbradley/csdat/big_covid/'

    runtags = 'run40 run41 run42 run43'.split()
    vecs_files = [bcdir+x for x in '''
round4_v3b_300_process_v1_leiden2_gex_pcs.npy
round4_v3b_400_process_v1_leiden2_gex_pcs.npy
round4_v3b_500_process_v1_leiden2_gex_pcs.npy
round4_v3b_run69_208_process_v1_leiden2_gex_pcs.npy
    '''.split() if x]
    expected_vecs_shape = (662874, 20)
    assert len(runtags) == len(vecs_files)

    for runtag,vecs_file in zip(runtags, vecs_files):

        rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
        if not exists(rundir):
            mkdir(rundir)

        cmds_file = f'{rundir}{runtag}_commands.txt'
        assert not exists(cmds_file)
        out = open(cmds_file,'w')

        print('reading:', vecs_file)
        vecs = np.load(vecs_file)
        assert vecs.shape == expected_vecs_shape
        num_tcrs = vecs.shape[0]

        num_jobs = (num_tcrs-1)//job_size+1

        xxargs = f' --num_nbrs {num_nbrs} '

        for job in range(num_jobs):
            start = job*job_size
            stop = (job+1)*job_size

            outfile_prefix = f'{rundir}{runtag}_{job}'
            cmd = (f'{PY} {EXE} {xargs} {xxargs} --mode {mode} '
                   f' --filename {vecs_file} '
                   f' --start_index {start} --stop_index {stop} '
                   f' --outfile_prefix {outfile_prefix} '
                   f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
            out.write(cmd+'\n')
        out.close()
        print('made:', cmds_file)
    exit()



if 0: # metaconga step ??
    # setup to average tcr aa vecs over GEX neighbors
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    job_size = 10000
    xargs = ' --sleeptime 30 '
    mode = 'vector_knns_average'
    num_nbrs = 5001
    mask_filename = None

    #runtag = 'run86' # KNN w/ GEX pcs for round8_v4cd4, avg cdr3 aas
    #runtag = 'run86m' # KNN w/ GEX pcs for round8_v4cd4, avg cdr3 aas, clump-MASKED!
    #runtag = 'run88' # KNN w/ GEX pcs for round8_v4cd4, avg v/j genes
    #runtag = 'run88m' # KNN w/ GEX pcs for round8_v4cd4, avg v/j genes
    #runtag = 'run90' # KNN w/ GEX pcs for round8_v4cd4, avg tcr features
    #runtag = 'run90m' # KNN w/ GEX pcs for round8_v4cd4, avg tcr features
    #runtag = 'run82' # KNN w/ GEX pcs for round8_v4cd4, avg gex vecs, not run yet
    #runtag = 'run94' # KNN w/ raw genes, clone size, and is_clumped
    #runtag = 'run94m' # KNN w/ raw genes, clone size, and is_clumped
    #runtag = 'run102' # KNN w/ more genes
    runtag = 'run102m' # KNN w/ more genes
    expected_vecs_shape = (2599503, 20)
    vecs_file     = '/home/pbradley/csdat/big_covid/round8_v5cd4_run84_xribo_200_process_v1_gex_pcs.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd4_run84_xribo_200_process_v1_leiden2_obs_tcr_aas.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd4_run84_xribo_200_process_v1_leiden2_obs_tcr_genes.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd4_run84_xribo_200_process_v1_tcr_features_X_features.npy'
    #avg_vecs_file = vecs_file
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd4_run84_xribo_200_process_v1_leiden2_raw_X_clone_size_clumped.npy'
    avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd4_run84_xribo_200_process_v1_leiden2_exgenes_X.npy'
    mask_filename = '/home/pbradley/csdat/big_covid/round8_v5cd4_run84_xribo_200_process_v1_leiden2_is_sz5_unclumped.npy'


    #runtag = 'run85' # KNN w/ GEX pcs for round8_v4cd8, avg cdr3 aas
    #runtag = 'run85m' # KNN w/ GEX pcs for round8_v4cd8, avg cdr3 aas, w/ MASK!
    #runtag = 'run87' # KNN w/ GEX pcs for round8_v4cd8, avg v/j genes
    #runtag = 'run87m' # KNN w/ GEX pcs for round8_v4cd8, avg v/j genes
    #runtag = 'run89' # KNN w/ GEX pcs for round8_v4cd8, avg tcr features
    #runtag = 'run89m' # KNN w/ GEX pcs for round8_v4cd8, avg tcr features
    #runtag = 'run91' # KNN w/ GEX pcs for round8_v4cd8, avg gex pcs # NOT RUN YET!
    #runtag = 'run93' # KNN w/ raw genes, clone size, and is_clumped
    #runtag = 'run93m' # KNN w/ raw genes, clone size, and is_clumped
    #runtag = 'run101' # GEX KNN avg additional genes not in top200 hs set
    #runtag = 'run101m' # GEX KNN avg additional genes not in top200 hs set
    #expected_vecs_shape = (815535, 20)
    #vecs_file =     '/home/pbradley/csdat/big_covid/round8_v5cd8_run84_xribo_200_process_v1_leiden2_gex_pcs.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd8_run84_xribo_200_process_v1_leiden2_obs_tcr_aas.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd8_run84_xribo_200_process_v1_leiden2_obs_tcr_genes.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd8_run84_xribo_200_process_v1_tcr_features_X_features.npy'
    #avg_vecs_file = vecs_file
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd8_run84_xribo_200_process_v1_leiden2_raw_X_clone_size_clumped.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v5cd8_run84_xribo_200_process_v1_leiden2_exgenes_X.npy'
    #mask_filename = '/home/pbradley/csdat/big_covid/round8_v5cd8_run84_xribo_200_process_v1_leiden2_is_sz5_unclumped.npy'


    ################## round8_v4:

    # runtag = 'run76' # KNN w/ GEX pcs for round8_v4cd4, avg cdr3 aas
    #runtag = 'run78' # KNN w/ GEX pcs for round8_v4cd4, avg v/j genes
    #runtag = 'run80' # KNN w/ GEX pcs for round8_v4cd4, avg tcr features
    #runtag = 'run82' # KNN w/ GEX pcs for round8_v4cd4, avg gex vecs
    #runtag = 'run84' # KNN w/ raw genes, clone size, and is_clumped
    #expected_vecs_shape = (2676334, 20)
    #vecs_file     = '/home/pbradley/csdat/big_covid/round8_v4cd4_run84_xribo_200_process_v1_gex_pcs.npy'
    # avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v4cd4_run84_xribo_200_process_v1_leiden2_obs_tcr_aas.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v4cd4_run84_xribo_200_process_v1_leiden2_obs_tcr_genes.npy'
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v4cd4_run84_xribo_200_process_v1_tcr_features_X_features.npy'
    #avg_vecs_file = vecs_file
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v4cd4_run84_xribo_200_process_v1_leiden2_raw_X_clone_size_clumped.npy'


    # runtag = 'run75' # KNN w/ GEX pcs for round8_v4cd8, avg cdr3 aas
    # runtag = 'run77' # KNN w/ GEX pcs for round8_v4cd8, avg v/j genes
    # runtag = 'run79' # KNN w/ GEX pcs for round8_v4cd8, avg tcr features
    #runtag = 'run81' # KNN w/ GEX pcs for round8_v4cd8, avg gex pcs
    #runtag = 'run83' # KNN w/ raw genes, clone size, and is_clumped
    #expected_vecs_shape = (830120, 20)
    #vecs_file =     '/home/pbradley/csdat/big_covid/round8_v4cd8_run84_xribo_200_process_v1_leiden2_gex_pcs.npy'
    # # # avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v4cd8_run84_xribo_200_process_v1_leiden2_obs_tcr_aas.npy'
    # # avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v4cd8_run84_xribo_200_process_v1_leiden2_obs_tcr_genes.npy'
    # avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v4cd8_run84_xribo_200_process_v1_tcr_features_X_features.npy'
    #avg_vecs_file = vecs_file
    #avg_vecs_file = '/home/pbradley/csdat/big_covid/round8_v4cd8_run84_xribo_200_process_v1_leiden2_raw_X_clone_size_clumped.npy'








    ##################################################################################88
    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    print('reading:', vecs_file)
    vecs = np.load(vecs_file)
    assert vecs.shape == expected_vecs_shape
    print('reading:', avg_vecs_file)
    avg_vecs = np.load(avg_vecs_file)
    assert avg_vecs.shape[0] == vecs.shape[0]
    num_tcrs = vecs.shape[0]

    num_jobs = (num_tcrs-1)//job_size+1

    xxargs = f' --num_nbrs {num_nbrs} '
    if mask_filename:
        xxargs += f' --mask_filename {mask_filename} '

    for job in range(num_jobs):
        start = job*job_size
        stop = (job+1)*job_size

        outfile_prefix = f'{rundir}{runtag}_{job}'
        cmd = (f'{PY} {EXE} {xargs} {xxargs} --mode {mode} '
               f' --filename {vecs_file} --avg_filename {avg_vecs_file} '
               f' --start_index {start} --stop_index {stop} '
               f' --outfile_prefix {outfile_prefix} '
               f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
        out.write(cmd+'\n')
    out.close()
    print('made:', num_jobs, cmds_file)
    exit()



if 0: # metaconga step 19
    # setup for some knn searches from vector inputs (here metaconga GEX pcs)
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    job_size = 10000
    xargs = ' --sleeptime 30 '
    mode = 'vector_knns'
    num_nbrs = 501

    runtag = 'run96' # KNN w/ GEX pcs
    expected_vecs_shape = (2599503, 20)
    vecs_file='/home/pbradley/csdat/big_covid/round8_v5cd4_run84_xribo_200_process_v1_gex_pcs.npy'

    # runtag = 'run95' # KNN w/ GEX pcs
    # expected_vecs_shape = (815535, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round8_v5cd8_run84_xribo_200_process_v1_gex_pcs.npy'

    # runtag = 'run72' # KNN w/ GEX pcs for round8_v4cd4
    # expected_vecs_shape = (2676334, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round8_v4cd4_run84_xribo_200_process_v1_gex_pcs.npy'

    # runtag = 'run69' # KNN w/ GEX pcs for round8_v4cd8
    # expected_vecs_shape = (830120, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round8_v4cd8_run84_xribo_200_process_v1_leiden2_gex_pcs.npy'

    # runtag = 'run64' # KNN w/ GEX pcs for round8_v3cd4
    # expected_vecs_shape = (2775426, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round8_v3cd4_run84_xribo_200_process_v1_leiden2_gex_pcs.npy'

    # runtag = 'run63' # KNN w/ GEX pcs for round8_v3cd8
    # expected_vecs_shape = (947664, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round8_v3cd8_run84_xribo_200_process_v1_leiden2_gex_pcs.npy' #round8_v3cd8_run84_xribo_200_process_v1_gex_pcs.npy'

    # runtag = 'run62' # KNN w/ GEX pcs for round8
    # expected_vecs_shape = (3903971, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round8_merged_common_genes_run84_xribo_top200_hotspot_genes_process_v1_gex_pcs.npy'

    # runtag = 'run52' # KNN w/ GEX pcs for
    # expected_vecs_shape = (3818745, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round6_merged_common_genes_run75_xribo_top200_hotspot_genes_process_v1_gex_pcs.npy'

    # runtag = 'run51' # KNN w/ GEX pcs for
    # expected_vecs_shape = (816035, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round6_v5b_run75_xribo_200_process_v1_gex_pcs.npy'


    # runtag = 'run39' # KNN w/ GEX pcs for round4-combo, top200 hotspot genes
    # expected_vecs_shape = (647204, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round4_v3c_process_v1_leiden2_gex_pcs.npy'

    # runtag = 'run37' # KNN w/ GEX pcs for round4-combo, top200 hotspot genes
    # expected_vecs_shape = (2049073, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round4_v4b_process_v1_leiden2_gex_pcs.npy'
    # runtag = 'run36' # KNN w/ GEX pcs for round4-combo, top200 hotspot genes
    # expected_vecs_shape = (662874, 20)
    # vecs_file='/home/pbradley/csdat/big_covid/round4_v3b_process_v1_leiden2_gex_pcs.npy'

    # runtag = 'run33' # KNN w/ GEX pcs for round4-combo, top200 hotspot genes
    # expected_vecs_shape = (2165233, 20)
    # vecs_file= '/home/pbradley/csdat/big_covid/round4_v4_process_v1_leiden2_gex_pcs.npy'

    # runtag = 'run31' # KNN w/ GEX pcs for round4-combo, top200 hotspot genes
    # expected_vecs_shape = (739702, 20)
    # vecs_file= '/home/pbradley/csdat/big_covid/round4_v3_process_v1_leiden2_gex_pcs.npy'

    # runtag = 'run30' # KNN w/ GEX pcs for round4-combo, top200 hotspot genes
    # expected_vecs_shape = (2895305, 20)
    #vecs_file='/home/pbradley/csdat/big_covid/round4_v2_process_v1_leiden2_gex_pcs.npy'

    # runtag = 'run27' # KNN w/ GEX pcs for round4-combo, top200 hotspot genes
    # expected_vecs_shape = (3064907, 20)
    #vecs_file= '/home/pbradley/csdat/big_covid/round4_combo_hotspot_top200_gex_pcs.npy'

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    print('reading:', vecs_file)
    vecs = np.load(vecs_file)
    assert vecs.shape == expected_vecs_shape
    num_tcrs = vecs.shape[0]

    num_jobs = (num_tcrs-1)//job_size+1

    xxargs = f' --num_nbrs {num_nbrs} '

    for job in range(num_jobs):
        start = job*job_size
        stop = (job+1)*job_size

        outfile_prefix = f'{rundir}{runtag}_{job}'
        cmd = (f'{PY} {EXE} {xargs} {xxargs} --mode {mode} '
               f' --filename {vecs_file} '
               f' --start_index {start} --stop_index {stop} '
               f' --outfile_prefix {outfile_prefix} '
               f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
        out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)
    exit()


if 0: # setup for some range and knn searches
    # clumping step 0
    # metaconga step 7
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    xargs = ' --sleeptime 30 '
    aa_mds_dim = 8
    job_size = 10000

    # round8 for clumping (redo)
    mode = 'paired_ranges'
    radius = 96.5
    bigfile = '/home/pbradley/csdat/koelle/zenaim/all_uniq_tcrs.tsv'
    filtfile = f'{DATADIR}phil/zenaim_filt.tsv'
    runtag = 'run105' ; job_size = 30000

    # # round8 for clumping (redo)
    # mode = 'paired_ranges'
    # radius = 96.5
    # bigfile = None
    # filtfile = f'{DATADIR}phil/big_combo_tcrs_2024-02-02_tsang.tsv'
    # runtag = 'run103'

    # round8 for clumping (redo)
    # mode = 'paired_ranges'
    # radius = 96.5
    # bigfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2024-02-02.tsv'
    # filtfile = f'{DATADIR}phil/big_combo_tcrs_2024-02-02_filt.tsv'
    # runtag = 'run59'


    # range searching for new style g-v-g
    #filtfile = '/home/pbradley/csdat/big_covid/round8_v5cd4_filt_tcrs.tsv'
    #filtfile = '/home/pbradley/csdat/big_covid/round8_v5cd8_filt_tcrs.tsv'
    #runtag = 'run99' # v5cd8
    #runtag = 'run100' # v5cd4
    #mode = 'paired_knns'
    #num_nbrs = 501
    #runtag = 'run97' # v5cd8
    #runtag = 'run98' # v5cd4
    #mode = 'paired_ranges'
    #radius = 120.5

    # # range searching for new style g-v-g
    # filtfile = '/home/pbradley/csdat/big_covid/round8_v4cd4_filt_tcrs.tsv'
    # #runtag = 'run73' # v4cd4
    # #mode = 'paired_knns'
    # #num_nbrs = 501
    # runtag = 'run74' # v4cd4
    # mode = 'paired_ranges'
    # radius = 120.5

    # filtfile = '/home/pbradley/csdat/big_covid/round8_v4cd8_filt_tcrs.tsv'
    # runtag = 'run71' # v4cd8
    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run70' # v4cd8
    # mode = 'paired_ranges'
    # radius = 120.5
    # filtfile = '/home/pbradley/csdat/big_covid/round8_v3cd4_filt_tcrs.tsv'
    # runtag = 'run68' # v3cd4
    #filtfile = '/home/pbradley/csdat/big_covid/round8_v3cd8_filt_tcrs.tsv'
    #runtag = 'run67' # v3cd8

    # # KNN TCR neighbors to prep for hotspot analysis
    # filtfile = '/home/pbradley/csdat/big_covid/round8_v3cd4_filt_tcrs.tsv'
    # runtag = 'run66' # v3cd4
    # # filtfile = '/home/pbradley/csdat/big_covid/round8_v3cd8_filt_tcrs.tsv'
    # # runtag = 'run65' # v3cd8
    # mode = 'paired_knns'
    # num_nbrs = 501

    # KNN TCR neighbors to prep for hotspot analysis
    # filtfile = f'{DATADIR}phil/round8_merged_filt.tsv'
    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run60' # redo with v3 fixed Terekhova; TCR KNN
    # runtag = 'run58' # redo with fixed Terekhova; TCR KNN
    #runtag = 'run57' # TCR KNN w/ round8_merged_filt.tsv

    # # round8 for clumping
    # mode = 'paired_ranges'
    # radius = 96.5
    # bigfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2024-01-18.tsv'
    # filtfile = f'{DATADIR}phil/big_combo_tcrs_2024-01-18_filt.tsv'
    # runtag = 'run55'

    # # round7 for clumping
    # mode = 'paired_ranges'
    # radius = 96.5
    # bigfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-12-17.tsv'
    # filtfile = f'{DATADIR}phil/big_combo_tcrs_2023-12-17_filt.tsv'
    # runtag = 'run53'


    # #bigfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-11-22.tsv'
    # filtfile = f'{DATADIR}phil/round6_merged_filt.tsv'
    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run50' # TCR KNN w/ round6_merged_filt.tsv


    # mode = 'paired_ranges'
    # radius = 96.5
    # bigfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-11-22.tsv'
    # filtfile = f'{DATADIR}phil/big_combo_tcrs_2023-11-22_filt.tsv'
    # runtag = 'run48'

    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run38'
    # filtfile = '/home/pbradley/csdat/big_covid/round4_v3c_filt_tcrs.tsv'

    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run35'
    # filtfile = '/home/pbradley/csdat/big_covid/round4_v4b_filt_tcrs.tsv'

    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run34'
    # filtfile = '/home/pbradley/csdat/big_covid/round4_v3b_filt_tcrs.tsv'

    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run32' # KNN w/ round4_v4_filt_tcrs.tsv
    # filtfile = '/home/pbradley/csdat/big_covid/round4_v4_filt_tcrs.tsv'

    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run29' # KNN w/ round4_v3_filt_tcrs.tsv

    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run28' # KNN w/ round4_v2_filt_tcrs.tsv

    # mode = 'paired_knns'
    # num_nbrs = 501
    # runtag = 'run26' # KNN w/ round4_merged_filt.tsv

    # mode = 'paired_knns'
    # num_nbrs = 101
    # runtag = 'run25' # KNN w/ round4_merged_filt.tsv

    # mode = 'paired_knns'
    # num_nbrs = 10
    # runtag = 'run24' # KNN w/ round4_merged_filt.tsv

    # mode = 'paired_knns'
    # num_nbrs = 25
    # runtag = 'run23' # KNN w/ round4_merged_filt.tsv

    #mode = 'paired_ranges'
    #radius = 96.5
    #runtag = 'run22' # w/ round4_merged_filt.tsv
    #runtag = 'run20' # w/ paired_sample_filt.tsv

    # bigfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-03-07.tsv'
    # filtfile = f'{DATADIR}phil/paired_sample_filt.tsv'
    # filtfile = f'{DATADIR}phil/round4_merged_filt.tsv'
    # filtfile = '/home/pbradley/csdat/big_covid/round4_v3_filt_tcrs.tsv'
    # filtfile = '/home/pbradley/csdat/big_covid/round4_v2_filt_tcrs.tsv'

    if not exists(filtfile):
        #assert False
        print('reading:', bigfile)
        tcrs = pd.read_table(bigfile)
        print('read:', tcrs.shape[0], bigfile)
        tcrs['cdr3a_nucseq'] = tcrs.cdr3a_nucseq.str.lower()
        tcrs['cdr3b_nucseq'] = tcrs.cdr3b_nucseq.str.lower()
        organism = 'human'
        tcrs = filter_out_bad_genes_and_cdr3s(
            tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
        tcrs = filter_out_bad_genes_and_cdr3s(
            tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')
        tcrs.to_csv(filtfile, sep='\t', index=False)
        print('made:', filtfile)

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    print('reading:', filtfile)
    tcrs = pd.read_table(filtfile, low_memory=False)
    num_tcrs = tcrs.shape[0]

    if 2: # sanity testing ##############
        organism = 'human'
        newtcrs = tcrs.copy()
        newtcrs = filter_out_bad_genes_and_cdr3s(
            newtcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
        newtcrs = filter_out_bad_genes_and_cdr3s(
            newtcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')
        print('filtfile sanity check:', newtcrs.shape[0], tcrs.shape[0])
        assert newtcrs.shape[0] == tcrs.shape[0]

    num_jobs = (num_tcrs-1)//job_size+1

    if mode == 'paired_ranges':
        xxargs = f' --radius {radius} '
    else:
        xxargs = f' --num_nbrs {num_nbrs} '

    for job in range(num_jobs):
        start = job*job_size
        stop = (job+1)*job_size

        outfile_prefix = f'{rundir}{runtag}_{job}'
        cmd = (f'{PY} {EXE} {xargs} {xxargs} --mode {mode} '
               f' --filename {filtfile} '
               f' --start_index {start} --stop_index {stop} '
               f' --aa_mds_dim {aa_mds_dim} '
               f' --outfile_prefix {outfile_prefix} '
               f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
        out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)
    exit()


if 0: # look at some paired data
    import faiss
    from pynndescent import NNDescent
    import tcrdist
    from tcrdist.tcr_sampler import parse_tcr_junctions#, resample_shuffled_tcr_chains

    aa_mds_dim = 8
    num_pos_cdr3 = 16

    bigfile = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2023-03-07.tsv'
    #subnum = 5000000
    #subnum = 500000
    subnum = 250000
    subfile = f'{DATADIR}phil/paired_sample_{subnum}.tsv'

    outprefix = subfile[:-4]+'_test'


    if not exists(subfile):
        print('reading:', bigfile)
        tcrs = pd.read_table(bigfile)
        print('read:', tcrs.shape[0], bigfile)
        tcrs['cdr3a_nucseq'] = tcrs.cdr3a_nucseq.str.lower()
        tcrs['cdr3b_nucseq'] = tcrs.cdr3b_nucseq.str.lower()
        if subnum>tcrs.shape[0]:
            tcrs.to_csv(subfile, sep='\t', index=False)
        else:
            tcrs.sample(subnum, random_state=11).to_csv(subfile, sep='\t', index=False)
        print('made:', subfile)


    organism = 'human'
    tcrs = pd.read_table(subfile)
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')

    fg_avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    fg_bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    if 0: # compute nearest neighbors in fg paired vecs
        fg_vecs = np.hstack([fg_avecs, fg_bvecs])

        num_nbrs = 20
        print('start NNDescent', num_nbrs, fg_vecs.shape)
        start = timer()
        index = NNDescent(
            fg_vecs, n_neighbors=num_nbrs, diversify_prob=1.0,
            pruning_degree_multiplier=1.5,
        )
        print(f'NNDescent took {timer()-start:.3f}', flush=True)
        I1, D1 = index.neighbor_graph
        np.save('I1.npy', I1)
        np.save('D1.npy', D1)

        print('start IndexFlatL2 knn search', num_nbrs, fg_vecs.shape)
        start = timer()
        idx = faiss.IndexFlatL2(fg_vecs.shape[1])
        idx.add(fg_vecs)
        D2, I2 = idx.search(fg_vecs, num_nbrs)
        np.save('I2.npy', I2)
        np.save('D2.npy', D2)
        print(f'IndexFlatL2 took {timer()-start:.2f}')

        exit()

    atcrs = tcrs['va ja cdr3a cdr3a_nucseq'.split()].itertuples(name=None, index=None)
    btcrs = tcrs['vb jb cdr3b cdr3b_nucseq'.split()].itertuples(name=None, index=None)
    junctions = parse_tcr_junctions('human', list(zip(atcrs, btcrs)))
    achains = resample_background_tcrs_v4(
        organism, 'A', junctions, preserve_vj_pairings=True)
    bg_atcrs = pd.DataFrame([dict(va=x[0], cdr3a=x[2]) for x in achains])
    bg_atcrs.to_csv(f'{outprefix}_bg_atcrs.tsv', sep='\t', index=False)
    bg_avecs = gapped_encode_tcr_chains(
        bg_atcrs, organism, 'A', aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)
    bchains = resample_background_tcrs_v4(
        organism, 'B', junctions, preserve_vj_pairings=False)
    bg_btcrs = pd.DataFrame([dict(vb=x[0], cdr3b=x[2]) for x in bchains])
    bg_btcrs.to_csv(f'{outprefix}_bg_btcrs.tsv', sep='\t', index=False)
    bg_bvecs = gapped_encode_tcr_chains(
        bg_btcrs, organism, 'B', aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)


    maxdist = 96
    n_fg = 50000
    n_bg = 50000
    abcounts = compute_background_paired_tcrdist_distributions(
        fg_avecs[:n_fg], fg_bvecs[:n_fg],
        bg_avecs[:n_bg], bg_bvecs[:n_bg], maxdist,
    )
    print('abcounts:', abcounts.mean(axis=0))
    np.save(outprefix+'_abcounts.npy', abcounts)
    exit()

    acounts = compute_background_single_tcrdist_distributions(
        fg_avecs, bg_avecs, maxdist)

    print('acounts:', acounts.mean(axis=0))
    np.save(outprefix+'_acounts.npy', acounts)

    bcounts = compute_background_single_tcrdist_distributions(
        fg_bvecs, bg_bvecs, maxdist)
    print('bcounts:', bcounts.mean(axis=0))
    np.save(outprefix+'_bcounts.npy', bcounts)


    if 0:
        idx = faiss.IndexFlatL2(fg_avecs.shape[1])
        idx.add(bg_avecs)
        start = timer()
        print('full avecs range search:', fg_avecs.shape[0], bg_avecs.shape[0])
        alims,aD,aI = idx.range_search(fg_avecs, 12.5)
        print(f'took {timer()-start:.2f}')

        idx = faiss.IndexFlatL2(fg_bvecs.shape[1])
        idx.add(bg_bvecs)
        print('full bvecs range search:', fg_bvecs.shape[0], bg_bvecs.shape[0])
        blims,bD,bI = idx.range_search(fg_bvecs, 12.5)
        print(f'took {timer()-start:.2f}')



    # for ii in range(10):
    #     print('run range search')
    #     print(f'done: {timer()-start:.2f}')
    # idx = faiss.IndexFlatL2(fg_avecs.shape[1])
    # idx.add(bg_bvecs)
    # start = timer()
    # print('run range search')
    # blims,bD,bI = idx.range_search(fg_bvecs, 96.5) ## this was super-slow
    # print(f'done: {timer()-start:.2f}')


    exit()





    if 0: # make some files for C++ testing
        # this didn't work: the C++ distance calculations were much slower than
        # faiss distances...
        cdr3_weight = 3.0

        chain = 'B'
        aa_vectors = calc_tcrdist_aa_vectors(aa_mds_dim, SQRT=True)
        gene_cdr_strings = setup_gene_cdr_strings(organism, chain)
        num_pos_other_cdrs = len(next(iter(gene_cdr_strings.values())))
        assert all(len(x)==num_pos_other_cdrs for x in gene_cdr_strings.values())

        seq_len = num_pos_other_cdrs + num_pos_cdr3
        vec_len = aa_mds_dim * seq_len
        print('gapped_encode_tcr_chains: aa_mds_dim=', aa_mds_dim,
              'num_pos_other_cdrs=', num_pos_other_cdrs,
              'num_pos_cdr3=', num_pos_cdr3, 'vec_len=', vec_len)

        vecs = []
        for v, cdr3 in zip(tcrs.vb, tcrs.cdr3b):
            v_vec = encode_sequence(gene_cdr_strings[v], aa_vectors)
            cdr3_vec = np.sqrt(cdr3_weight) * gapped_encode_cdr3(
                cdr3, aa_vectors, num_pos_cdr3)

        outfile = 'vec_tcr_seqs.txt'
        out = open(outfile, 'w')
        for v, cdr3 in zip(tcrs.vb, tcrs.cdr3b):
            vgapped = gene_cdr_strings[v]
            cdr3gapped = trim_and_gap_cdr3(cdr3, num_pos_cdr3)
            assert len(vgapped+cdr3gapped) == seq_len
            out.write(vgapped + cdr3gapped + '\n')
        out.close()
        print('made:', tcrs.shape[0], outfile)

        outfile = 'vec_tcr_vecs.txt'
        out = open(outfile,'w')
        out.write(f'aa_dim {aa_mds_dim}\nseq_len {seq_len}\nseq_wts')
        for i in range(seq_len):
            wt = 1.0 if i < num_pos_other_cdrs else 3.0
            out.write(f' {wt}')
        out.write('\n')
        for aa,vecs in aa_vectors.items():
            out.write(aa+' '+(' '.join(str(x) for x in vecs))+'\n')
        out.close()
        print('made:', outfile)
        out.close()
        exit()




if 0: # compare our hits to ALICE hits in terms of overlap with YFV-expanding clones
    import tcrdist

    timepoint = 15
    bgnum = 6
    radius = 12.5

    tcrdister = tcrdist.tcr_distances.TcrDistCalculator('human')
    alice_tcrs = read_yfv_alice_hits()
    xclones = read_yfv_expanded_clones()

    donors = set(xclones.donor)

    for donor in donors:
        amask = (alice_tcrs.donor==donor)&(alice_tcrs.timepoint==timepoint)
        atcrs = alice_tcrs[amask].drop_duplicates('v cdr3aa'.split())
        xtcrs = xclones[xclones.donor==donor].drop_duplicates('v cdr3aa'.split())
        num_alice_hits = atcrs.shape[0]
        num_xtcrs = xtcrs.shape[0]
        D = calc_tcrdist_matrix(atcrs, xtcrs, tcrdister)
        threshold = 12.5
        a_overlap = np.sum(D.min(axis=1)<threshold)
        x_overlap = np.sum(D.min(axis=0)<threshold)
        print('overlap: alice', donor, a_overlap/num_alice_hits, x_overlap/num_xtcrs,
              flush=True)

        # read the tcrs
        ftag = f'{donor}_{timepoint}_F1__min_count_2.txt'
        tcrs = read_britanova_tcrs(
            f'/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/{ftag}.gz')

        # read the nbr counts
        runtag = 'run9' if bgnum <6 else 'run9b' if bgnum==6 else 'run9c'
        prefix = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/{runtag}'
        #run9b_Q2_15_F1__min_count_2.txt_12.5_r0_fg_nbr_counts.npy
        fg_counts=np.load(f'{prefix}_{ftag}_{radius:.1f}_r0_fg_nbr_counts.npy')
        num_tcrs = fg_counts.shape[0]
        assert num_tcrs == tcrs.shape[0]

        # read bg counts, compute pvals
        bg_counts = np.zeros((num_tcrs,))
        for r in range(10):
            bg_counts += np.load(f'{prefix}_{ftag}_{radius:.1f}_r{r}_bg_'
                                 f'{bgnum}_nbr_counts.npy')
        num_bg_tcrs = 10*num_tcrs


        min_fg_bg_nbr_ratio = 2. # actually these are the function defaults
        max_fg_bg_nbr_ratio = 100
        pvals = compute_nbr_count_pvalues(
            fg_counts, bg_counts, num_bg_tcrs,
            min_fg_bg_nbr_ratio=min_fg_bg_nbr_ratio,
            max_fg_bg_nbr_ratio=max_fg_bg_nbr_ratio,
        )

        pvals = pvals.join(tcrs['count v j cdr3nt cdr3aa'.split()].reset_index(),
                           on='tcr_index')
        pvals = pvals.sort_values('evalue').drop_duplicates(['v','cdr3aa'])

        pvals = pvals.head(num_alice_hits)
        assert pvals.shape[0] == num_alice_hits

        D = calc_tcrdist_matrix(pvals, xtcrs, tcrdister)
        a_overlap = np.sum(D.min(axis=1)<threshold)
        x_overlap = np.sum(D.min(axis=0)<threshold)
        print('overlap pvals', donor, a_overlap/num_alice_hits, x_overlap/num_xtcrs,
              'max_evalue:', pvals.evalue.iloc[-1], flush=True)


    exit()

if 0: # clean igor tcrs
    # in the end, this just removed 155 short cdr3s
    fname = 'data/phil/big_background.tsv'
    new_fname = 'data/phil/big_background_filt.tsv'

    tcrs = pd.read_table(fname).rename(
        columns={'junction':'cdr3nt', 'junction_aa':'cdr3aa',
                 'v_call':'v', 'j_call':'j'})

    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, v_column, cdr3_column, organism, chain, j_column=j_column)
    print('num_tcrs:', tcrs.shape[0], fname)
    tcrs.to_csv(new_fname, sep='\t', index=False)
    print('made:', new_fname)
    exit()


if 0: # plot pval test
    import seaborn as sns

    df = pd.read_table('pvaltest_100000_1000000_24.5_100.tsv')

    fg_size = 1e5

    plt.figure()
    colname = 'num_pvalues_lt_threshold'
    df[colname] = np.log10(1+df.num_better)
    sns.boxplot(data=df, x='threshold', y=colname, color='gray')
    locs = plt.xticks()[0]

    num_w_nbrs = df.num_with_ge_1_nbr.mean()
    thresholds = np.sort(df.threshold.unique())
    exp_vals_hi = np.log10(1+fg_size*thresholds)
    exp_vals_lo = np.log10(1+num_w_nbrs*thresholds)
    plt.plot(locs, exp_vals_hi, ':r', label='expected based on fg size')
    plt.plot(locs, exp_vals_lo, ':g', label='expected based on num tcrs w/ any fg nbrs')
    plt.legend()
    locs = np.array([0, 1, 10, 100, 1000, 10000])
    labels = [str(x) for x in locs]
    locs = np.log10(1+locs)
    # locs =plt.yticks()[0]
    # labels = [f'{10**x - 1:.2f}' for x in locs]
    plt.yticks(locs, labels)
    plt.xlabel('Hypergeometric P-value threshold')
    plt.title('fg_size= 1e5, bg_size= 1e6, radius= 24.5, num_repeats= 100\n'
              'aa_mds_dim= 8, fg and bg drawn from IGOR-sampled repertoire',
              fontsize=10)
    plt.tight_layout()
    plt.savefig('/home/pbradley/csdat/raptcr/tmp.png')
    exit()

if 0: # simple test of hypergeom pvals on random repertoires

    fname = 'data/phil/big_background_2e6.tsv'
    #fname = 'data/phil/big_background_1e6.tsv'
    #fname = 'data/phil/big_background.tsv'
    big_tcrs = pd.read_table(fname).rename(
        columns={'junction':'cdr3nt', 'junction_aa':'cdr3aa',
                 'v_call':'v', 'j_call':'j'})

    num_repeats = 100

    fg_size = 100000
    bg_size = 1000000
    radius = 24.5

    outfile = f'pvaltest_{fg_size}_{bg_size}_{radius:.1f}_{num_repeats}.tsv'

    thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    dfl = []
    for r in range(num_repeats):
        tcrs = big_tcrs.sample(fg_size+bg_size)

        fg_tcrs = tcrs.head(fg_size)
        bg_tcrs = tcrs.tail(bg_size)

        v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
        aa_mds_dim = 8
        fg_vecs = gapped_encode_tcr_chains(
            fg_tcrs, organism, chain, aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column).astype(np.float32)

        bg_vecs = gapped_encode_tcr_chains(
            bg_tcrs, organism, chain, aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column).astype(np.float32)

        fg_counts = compute_nbr_counts_flat(fg_vecs, fg_vecs, radius)-1
        mask = fg_counts>=1
        mask_bg_counts = compute_nbr_counts_flat(fg_vecs[mask], bg_vecs, radius)
        bg_counts = np.empty_like(fg_counts)
        bg_counts[mask] = mask_bg_counts
        bg_counts[~mask] = 1000

        min_fg_bg_nbr_ratio = 0 # everything!
        max_fg_bg_nbr_ratio = 100 # not used for pvals
        target_bg_nbrs = None

        pvals = compute_nbr_count_pvalues(
            fg_counts, bg_counts, bg_size,
            min_fg_nbrs=1,
            min_fg_bg_nbr_ratio=min_fg_bg_nbr_ratio,
            max_fg_bg_nbr_ratio=max_fg_bg_nbr_ratio,
            target_bg_nbrs=target_bg_nbrs,
        )

        for t in thresholds:
            count = (pvals.pvalue<=t).sum()
            print(f'{r:2d} t= {t:9.2e} obs: {count} exp: {t*fg_size:.3f}', flush=True)
            dfl.append(dict(
                threshold=t,
                num_better=count,
                expected_num_better=fg_size*t,
                num_with_ge_1_nbr=mask.sum(),
                repeat=r,
                fg_size=fg_size,
                bg_size=bg_size,
            ))

        pd.DataFrame(dfl).to_csv(outfile, sep='\t', index=False)
    pd.DataFrame(dfl).to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

    exit()



if 0: # miscellaneous code for visualizing the top significant tcrs in the emerson set
    from scipy.stats import hypergeom
    import faiss


    if 2: # make graph plot of the tcrs
        import networkx as nx
        import matplotlib.pyplot as plt

        min_hipids_count = 20
        outprefix=f'/home/pbradley/csdat/raptcr/run17run18_umap_mc_{min_hipids_count}'

        tcrs = pd.read_table('run17run18_hla_assoc_pvals_input_tcrs_pvals.tsv')
        print('done reading')
        graph_tcrs = tcrs[tcrs.tcr_aa_hipids_count>=min_hipids_count]\
                     .drop_duplicates('tcr_aa')
        print('done mask+deduping')
        if min_hipids_count==50:
            assert graph_tcrs.shape[0] == 527 # hack
        graph_tcrs.sort_values('tcr_aa_hipids_count', inplace=True, ascending=False)

        v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
        aa_mds_dim = 8
        #graph_tcrs = pd.read_table('tmp.tsv')
        vecs = gapped_encode_tcr_chains(
            graph_tcrs, organism, chain, aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column).astype(np.float32)


        hla_pvals = pd.read_table('run17run18_hla_assoc_pvals.tsv')
        hla_pvals = hla_pvals.sort_values('pvalue').drop_duplicates('tcr_aa').\
                    set_index('tcr_aa', drop=False)
        print('hla_pvals:', hla_pvals.allele.value_counts())

        graph_tcrs = graph_tcrs.join(hla_pvals['pvalue allele'.split()],
                                     on='tcr_aa', rsuffix='_hla')
        graph_tcrs['pvalue_hla'] = graph_tcrs.pvalue_hla.fillna(1.0)


        if 2:
            import umap
            reducer = umap.UMAP(random_state=11, min_dist=1)
            print('run UMAP:', vecs.shape)
            xy = reducer.fit_transform(vecs)
            graph_tcrs['umap_0'] = xy[:,0]
            graph_tcrs['umap_1'] = xy[:,1]
            print('done')

            clusters = compute_leiden_clusters_from_vecs(vecs)
            graph_tcrs['leiden'] = clusters
            print('num clusters:', np.max(clusters)+1)
            cmap = plt.get_cmap('tab20')
            cluster_colors = [cmap.colors[i%20] for i in clusters]

            plt.figure(figsize=(12,12))
            sizes = graph_tcrs.tcr_aa_hipids_count * 20/50.
            plt.scatter(xy[:,0], xy[:,1], c=cluster_colors, s=sizes)#20)
            for c in range(np.max(clusters)+1):
                row = graph_tcrs[graph_tcrs.leiden==c].iloc[0]
                plt.text(row.umap_0, row.umap_1,
                         row.v.split('*')[0]+'_'+row.cdr3aa[3:-2],
                         fontsize=6)

            # draw some edges
            idx = faiss.IndexFlatL2(vecs.shape[1])
            idx.add(vecs)
            start = timer()
            print('run range search')
            lims,D,I = idx.range_search(vecs, 12.5)
            print(f'vecs range_search took {timer()-start:.2f} seconds')
            for ii,(start,stop) in enumerate(zip(lims[:-1], lims[1:])):
                for jj in I[start:stop]:
                    plt.plot([xy[ii,0], xy[jj,0]], [xy[ii,1], xy[jj,1]],
                             c='k', zorder=-1, linewidth=.5)


            plt.tight_layout()
            pngfile = outprefix+'.png'
            plt.savefig(pngfile)
            print('made:', pngfile)


            # make another one colored by HLA pvalue
            colors = np.sqrt(-1*np.log10(graph_tcrs.pvalue_hla))
            plt.figure(figsize=(12,12))
            plt.scatter(xy[:,0], xy[:,1], c=colors, s=20, vmax=np.sqrt(25),
                        vmin=np.sqrt(3))
            #plt.colorbar()
            plt.tight_layout()
            pngfile = outprefix+'_hla_pvals.png'
            plt.savefig(pngfile)
            print('made:', pngfile)


            exit()

        idx = faiss.IndexFlatL2(vecs.shape[1])
        idx.add(vecs)
        start = timer()
        print('run range search')
        lims,D,I = idx.range_search(vecs, 12.5)
        print(f'vecs range_search took {timer()-start:.2f} seconds')

        G = nx.Graph()
        for ii, (start,stop) in enumerate(zip(lims[:-1], lims[1:])):
            for nbr in I[start:stop]:
                if ii<nbr:
                    G.add_edge(ii,nbr)

        plt.figure(figsize=(12,12))

        nodes = set(G.nodes())

        labels = {ii:x for ii,x in enumerate(graph_tcrs.tcr_aa)
                  if ii in nodes}

        k = 2/np.sqrt(len(G.nodes())) # default is 1/sqrt(N)
        pos = nx.drawing.layout.spring_layout(G, k=k)

        nx.draw_networkx(G, pos, ax=plt.gca(), node_size=10, labels=labels)
                         #with_labels=False)
        plt.show()
        plt.close('all')


    exit()


if 0: # look for associations between significant tcrs and hla alleles on emerson set
    from scipy.stats import hypergeom
    #import faiss

    min_allele_count = 10
    min_tcr_count = 5
    num_hipids = 666
    min_overlap = 5
    outfile = 'run17run18_hla_assoc_pvals.tsv'

    fnames = glob('run1[78]_pvals.tsv')
    dfl = []
    for fname in fnames:
        print('reading:', fname)
        dfl.append(pd.read_table(fname))
        print('done')

    print('concatenating')
    pvals = pd.concat(dfl)
    print('DONE concatenating')
    hipids = sorted(pvals.hipid.unique())
    assert len(hipids) == num_hipids
    assert len(pvals.hipid.unique()) == num_hipids
    pvals['tcr_aa'] = pvals.v + "_" + pvals.cdr3aa
    pvals_occs = pvals['tcr_aa hipid'.split()].drop_duplicates()
    tcr_counts = pvals_occs.tcr_aa.value_counts()
    tcr_counts = tcr_counts[tcr_counts>=min_tcr_count]
    tcrs = list(tcr_counts.index)
    tcrs_set = set(tcr_counts.index)
    tcr_counts.name = 'tcr_aa_hipids_count'
    pvals = pvals[pvals.tcr_aa.isin(tcrs_set)].join(tcr_counts, on='tcr_aa')
    print('num_tcrs above count:', len(tcrs))

    outfile_pvals = outfile[:-4]+'_input_tcrs_pvals.tsv'
    pvals.to_csv(outfile_pvals, sep='\t', index=False)
    print('made:', outfile_pvals)

    tcr_occs = np.zeros((len(tcrs), num_hipids), dtype=bool)
    tcr2index = {x:i for i,x in enumerate(tcrs)}
    hipid2index = {x:i for i,x in enumerate(hipids)}
    print('setup tcr_occs')
    for l in pvals.itertuples():
        tcr_occs[tcr2index[l.tcr_aa], hipid2index[l.hipid]] = True
    print('done')

    fname = '/home/pbradley/csdat/raptcr/emerson/newest_hla_info_df.tsv'
    hla_df = pd.read_table(fname)

    loci = sorted(set([x[:-2] for x in hla_df.columns if x!='hipid']))

    dfl = []
    for locus in loci:
        col0,col1 = hla_df[locus+'_0'], hla_df[locus+'_1']
        locus_hipids = set(hla_df[~col0.isna()].hipid)
        locus_mask = np.array([x in locus_hipids for x in hipids])
        total_locus = locus_mask.sum()
        alleles = set(col0)|set(col1)
        print('total_locus:', total_locus, locus, flush=True)

        for allele in alleles:
            posmask = (col0==allele)|(col1==allele)
            if posmask.sum()<min_allele_count:
                continue
            allele_hipids = set(hla_df[posmask].hipid)
            allele_mask = np.array([x in allele_hipids for x in hipids])
            total_w_allele = allele_mask.sum()
            assert total_w_allele == posmask.sum()

            for itcr, tcr in enumerate(tcrs):
                tcr_mask = tcr_occs[itcr]
                overlap = np.sum(tcr_mask&allele_mask)
                if overlap >= min_overlap:
                    total_w_tcr = sum(tcr_mask&locus_mask)
                    pval = hypergeom.sf(
                        overlap-1, total_locus, total_w_tcr, total_w_allele)
                    if pval<1e-3:
                        print('pval:', pval, overlap, total_w_tcr, total_w_allele,
                              total_locus, tcr, allele)
                        dfl.append(dict(
                            pvalue=pval,
                            overlap=overlap,
                            total_w_tcr=total_w_tcr,
                            total_w_allele=total_w_allele,
                            total_locus=total_locus,
                            locus=locus,
                            allele=allele,
                            tcr_aa=tcr,
                            v = tcr.split('_')[0],
                            cdr3aa = tcr.split('_')[1],
                        ))


    pd.DataFrame(dfl).to_csv(outfile, sep='\t', index=False)

    exit()



if 0: # look at the emerson A*02:01 repertoire results; also britanova results

    runtag = 'run16'
    #runtag = 'run18'
    #runtag = 'run17'
    radii = [12.5, 18.5, 24.5]
    num_repeats = 10
    bg_num = 6
    max_evalue = 100.
    BRITANOVA = False

    outfile = f'{runtag}_pvals.tsv'

    if runtag == 'run18':
        fnames = glob('/home/pbradley/csdat/raptcr/emerson/*reparsed.tsv')
        assert len(fnames) == 666
        old_fnames = set(get_emerson_files_for_allele('A*02:01'))
        fnames = [x for x in fnames if x not in old_fnames]
        assert len(fnames) == 666 - 268
    elif runtag == 'run16':
        BRITANOVA = True
        fnames = glob('/home/pbradley/gitrepos/immune_response_detection/'
                      'data/phil/britanova/A*gz')
    else:
        fnames = get_emerson_files_for_allele('A*02:01')
    runprefix = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/{runtag}'

    dfl = []
    for fname in fnames:
        print(fnames.index(fname), len(fnames), fname, flush=True)
        ftag = fname.split('/')[-1][:-3]
        if BRITANOVA:
            hipid = ftag
        else:
            hipid = ftag[:8]
        tcrs = read_britanova_tcrs(fname, max_tcrs = 500000)
        for radius in radii:

            bg_files = glob(f'{runprefix}_{ftag}_{radius:.1f}_*_bg_{bg_num}*npy')
            print(ftag, 'numfiles:', len(bg_files), radius, fname)
            if len(bg_files) != num_repeats:
                print('ERROR missing:', len(bg_files), radius, fname)
                continue
            # read fg counts
            fg_counts=np.load(f'{runprefix}_{ftag}_{radius:.1f}_r0_fg_nbr_counts.npy')
            num_tcrs = fg_counts.shape[0]
            assert num_tcrs == tcrs.shape[0]

            # read bg counts, compute pvals
            bg_counts = np.zeros((num_tcrs,))
            for r in range(num_repeats):
                bg_counts += np.load(f'{runprefix}_{ftag}_{radius:.1f}_r{r}_bg_'
                                     f'{bg_num}_nbr_counts.npy')

            num_bg_tcrs = num_repeats*num_tcrs


            min_fg_bg_nbr_ratio = 2. # actually these are the function defaults
            max_fg_bg_nbr_ratio = 100
            target_bg_nbrs = None
            pvals = compute_nbr_count_pvalues(
                fg_counts, bg_counts, num_bg_tcrs,
                min_fg_bg_nbr_ratio=min_fg_bg_nbr_ratio,
                max_fg_bg_nbr_ratio=max_fg_bg_nbr_ratio,
                target_bg_nbrs=target_bg_nbrs,
            )
            pvals = pvals.join(tcrs['count v j cdr3nt cdr3aa'.split()].reset_index(),
                               on='tcr_index')
            pvals = pvals.sort_values('evalue').drop_duplicates(['v','j','cdr3aa'])
            pvals = pvals[pvals.evalue<= max_evalue]
            pvals['radius'] = radius
            pvals['hipid'] = hipid
            dfl.append(pvals)

        # make tmp results for tracking...
        if dfl:
            pd.concat(dfl).to_csv(outfile, sep='\t', index=False)


    if dfl:
        df = pd.concat(dfl)
        df.to_csv(outfile, sep='\t', index=False)
        print('made:', outfile)

    exit()


if 0: # testing mods to current favorite background model
    import tcrdist
    from tcrdist.tcr_sampler import parse_tcr_junctions, resample_shuffled_tcr_chains

    fname = DATADIR+'phil/britanova/A5-S18.txt.gz'
    #fname = './data/phil/britanova/A5-S11.txt.gz'
    tcrs = read_britanova_tcrs(fname)#.head(10000)

    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcr_tuples = [(None,x) for x in tcrs['v j cdr3aa cdr3nt'.split()].itertuples(
        name=None, index=None)]

    junctions = parse_tcr_junctions('human', tcr_tuples)

    res = resample_background_tcrs_v4('human','B',junctions)
    exit()


if 0: # make some background tcrs for paired data
    from tcrdist.tcr_sampler import (parse_tcr_junctions, resample_shuffled_tcr_chains,
                                     find_alternate_alleles_for_tcrs)
    FIX = True
    organism = 'human'

    fname = DATADIR+'phil/paired_sample_250000.tsv'
    #fname = DATADIR+'phil/paired_sample_50000.tsv'
    #v_column, j_column, cdr3_column, organism, chain= 'va','ja','cdr3a','human','A'
    #v_column, j_column, cdr3_column, organism, chain= 'vb','jb','cdr3b','human','B'
    tcrs = read_paired_data(fname)

    acols = 'va ja cdr3a cdr3a_nucseq'.split()
    bcols = 'vb jb cdr3b cdr3b_nucseq'.split()
    atcr_tuples = tcrs[acols].itertuples(name=None, index=None)
    btcr_tuples = tcrs[bcols].itertuples(name=None, index=None)

    if FIX:
        tcr_tuples_fixed = find_alternate_alleles_for_tcrs(
            organism, list(zip(atcr_tuples, btcr_tuples)), verbose=True)
        atcr_tuples =  [x[0] for x in tcr_tuples_fixed]
        btcr_tuples =  [x[1] for x in tcr_tuples_fixed]

    junctions = parse_tcr_junctions('human', list(zip(atcr_tuples, btcr_tuples)))

    bg_atcr_tuples = resample_background_tcrs_v4(
        organism, 'A', junctions, preserve_vj_pairings=True)
    bg_btcr_tuples = resample_background_tcrs_v4(
        organism, 'B', junctions, preserve_vj_pairings=False)

    bg_tcrs = pd.DataFrame([
        dict(va=va, ja=ja, cdr3a=cdr3a, cdr3a_nucseq=cdr3a_nucseq,
             vb=vb, jb=jb, cdr3b=cdr3b, cdr3b_nucseq=cdr3b_nucseq)
        for ((va,ja,cdr3a,cdr3a_nucseq),(vb,jb,cdr3b,cdr3b_nucseq)) in zip(
                bg_atcr_tuples, bg_btcr_tuples)])

    bg_tcrs = filter_out_bad_genes_and_cdr3s(
        bg_tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
    bg_tcrs = filter_out_bad_genes_and_cdr3s(
        bg_tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')

    num_tcrs = bg_tcrs.shape[0]
    print('final num tcrs:', num_tcrs)

    outfile = f'{DATADIR}phil/paired_bg_tcrs_{num_tcrs}.tsv'
    bg_tcrs.to_csv(outfile, sep='\t', index=False)
    exit()



if 0: # look at current favorite background model, seems to make too few 0-N cdr3s
    import tcrdist
    import itertools as it
    from tcrdist.tcr_sampler import (parse_tcr_junctions, resample_shuffled_tcr_chains,
                                     find_alternate_alleles_for_tcrs)

    LOG = True
    FIX = True

    if 2:
        fname = DATADIR+'phil/paired_sample_250000.tsv'
        #fname = DATADIR+'phil/paired_sample_50000.tsv'
        #v_column, j_column, cdr3_column, organism, chain= 'va','ja','cdr3a','human','A'
        v_column, j_column, cdr3_column, organism, chain= 'vb','jb','cdr3b','human','B'
        tcrs = read_paired_data(fname)

        acols = 'va ja cdr3a cdr3a_nucseq'.split()
        bcols = 'vb jb cdr3b cdr3b_nucseq'.split()
        atcr_tuples = tcrs[acols].itertuples(name=None, index=None)
        btcr_tuples = tcrs[bcols].itertuples(name=None, index=None)

        if FIX:
            tcr_tuples_fixed = find_alternate_alleles_for_tcrs(
                organism, list(zip(atcr_tuples, btcr_tuples)), verbose=True)
            atcr_tuples =  [x[0] for x in tcr_tuples_fixed]
            btcr_tuples =  [x[1] for x in tcr_tuples_fixed]

        ftag = fname.split('/')[-1][:-4]+'_'+chain
        if chain == 'A':
            tcr_tuples = list(zip(atcr_tuples, it.repeat(None)))
        else:
            tcr_tuples = list(zip(it.repeat(None), btcr_tuples))
    else:
        fname = DATADIR+'phil/britanova/A5-S18.txt.gz'
        #fname = './data/phil/britanova/A5-S11.txt.gz'
        ftag = fname.split('/')[-1][:-7]
        v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
        tcrs = read_britanova_tcrs(fname)

        tcr_tuples = [(None,x) for x in tcrs['v j cdr3aa cdr3nt'.split()].itertuples(
            name=None, index=None)]

    junctions = parse_tcr_junctions('human', tcr_tuples)

    fg_nucseq_srcs = list(junctions[f'cdr3{chain.lower()}_nucseq_src'])
    # remove '*0N' from gene names
    fg_genes = [(x.split('*')[0], y.split('*')[0]) # tuple: (v,j)
                for x,y in zip(junctions['v'+chain.lower()],
                               junctions['j'+chain.lower()])]
    v_genes = [x[0] for x in Counter([y[0] for y in fg_genes]).most_common()]
    j_genes = [x[0] for x in Counter([y[1] for y in fg_genes]).most_common()]
    genes_ordered = [v_genes, j_genes]

    nrows, ncols = 2,5
    plt.figure(figsize=(4*ncols,4*nrows))
    for row,bgnum in enumerate([4,6]):
        if bgnum==4:
            bg_tcr_tuples, src_junction_indices = resample_shuffled_tcr_chains(
                organism, tcrs.shape[0], chain, junctions,
                return_src_junction_indices=True,
            )
            bg_nucseq_srcs = []
            for tcr, inds in zip(bg_tcr_tuples, src_junction_indices):
                if len(bg_nucseq_srcs)%150000==0:
                    print(len(bg_nucseq_srcs))
                vseq = fg_nucseq_srcs[inds[0]]
                jseq = fg_nucseq_srcs[inds[1]]
                bg_nucseq_srcs.append(vseq[:inds[2]] + jseq[inds[2]:])
                assert len(tcr[3]) == len(bg_nucseq_srcs[-1])
        else:
            preserve_vj_pairings = (chain=='A')
            bg_tcr_tuples, bg_nucseq_srcs = resample_background_tcrs_v4(
                organism, chain, junctions, preserve_vj_pairings=preserve_vj_pairings,
                return_nucseq_srcs=True)
        bg_genes = [(x[0].split('*')[0], x[1].split('*')[0])
                    for x in bg_tcr_tuples]

        assert len(fg_nucseq_srcs) == len(bg_nucseq_srcs)
        for tag, srcs, genes in zip(['fg','bg'], [fg_nucseq_srcs, bg_nucseq_srcs],
                                    [fg_genes, bg_genes]):
            plt.subplot(nrows, ncols, 1+row*ncols)
            lens = [len(x)//3 for x in srcs]
            counts = Counter(lens)
            mn,mx = min(counts.keys()), max(counts.keys())
            mx = 25
            xvals = range(mn,mx+1)
            yvals = [counts[x] for x in xvals]
            if LOG:
                yvals = np.log1p(yvals)
            plt.plot(xvals, yvals, label=tag)
            plt.title(f'bgnum={bgnum} cdr3aa len')

            plt.subplot(nrows, ncols, 2+row*ncols)
            lens = [x.count('N') for x in srcs]
            counts = Counter(lens)
            mn,mx = min(counts.keys()), max(counts.keys())
            mx = 25
            xvals = range(mn,mx+1)
            yvals = [counts[x] for x in xvals]
            if LOG:
                yvals = np.log1p(yvals)
            plt.plot(xvals, yvals, label=tag)
            plt.title(f'bgnum={bgnum} num N nucs')
            print('N0:', tag, counts[0])

            plt.subplot(nrows, ncols, 3+row*ncols)
            lens = [x.count('D') for x in srcs]
            counts = Counter(lens)
            mn,mx = min(counts.keys()), max(counts.keys())
            xvals = range(mn,mx+1)
            yvals = [counts[x] for x in xvals]
            if LOG:
                yvals = np.log1p(yvals)
            plt.plot(xvals, yvals, label=tag)
            plt.title(f'bgnum={bgnum} num D nucs')

            for ig, vj in enumerate('vj'):
                plt.subplot(nrows, ncols, 4+ig+row*ncols)
                counts = Counter(x[ig] for x in genes)
                yvals = [counts[x] for x in genes_ordered[ig]]
                if LOG:
                    yvals = np.log1p(yvals)
                xvals = range(len(genes_ordered[ig]))
                plt.plot(xvals, yvals, label=tag)
                plt.title(f'bgnum={bgnum} {vj} gene counts')

    plt.legend()
    pngfile = f'/home/pbradley/csdat/raptcr/bg4and6_{ftag}_dists.png'
    if LOG:
        pngfile = pngfile[:-4]+'_log1p.png'
    if FIX:
        pngfile = pngfile[:-4]+'_fix.png'
    plt.savefig(pngfile)
    print('made:', pngfile)

    exit()

if 0: # read our parsed version of the emerson repertoires, make 'britanova'-like files
    #files = glob('/fh/fast/matsen_e/shared/tcr-gwas/emerson_stats/'
    #             'trim_stats_HIP00110_w_pnucs_and_ties.tsv')
    files = glob('/fh/fast/matsen_e/shared/tcr-gwas/emerson_stats/'
                 'trim_stats_HIP?????_w_pnucs_and_ties.tsv')
    assert len(files) == 666

    for fname in files:
        tcrs = read_reparsed_emerson_tcrs(fname)
        hipid = fname.split('_')[-5]
        outfile = f'/home/pbradley/csdat/raptcr/emerson/{hipid}_reparsed.tsv'
        tcrs.to_csv(outfile, sep='\t', index=False)
        print('made:', tcrs.shape[0], outfile, flush=True)

    exit()




if 0: # read, parse, and subset the YFV d0 and d15 tcrs
    min_count = 2
    yfvdir = '/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
    files = glob(yfvdir+'??_0_F?_.txt')
    assert len(files) == 12
    #files = glob(yfvdir+'??_15_F?_.txt')
    #assert len(files) == 7

    for fname in files:
        tcrs = read_yfv_tcrs(fname)
        outfile = f'{fname[:-4]}_min_count_{min_count}.txt'
        tcrs.to_csv(outfile, sep='\t', index=False)
        print('made:', outfile)

    exit()

if 0: # testing new background model; it's not that great!
    import tcrdist
    organism = 'human'

    fname = './data/phil/britanova/A5-S11.txt.gz'
    tcrs = read_britanova_tcrs(fname)

    tcr_tuples = [(None,x) for x in tcrs['v j cdr3aa cdr3nt'.split()].itertuples(
        name=None, index=None)]
    junctions = tcrdist.tcr_sampler.parse_tcr_junctions(organism, tcr_tuples)

    junctions = add_vdj_splits_info_to_junctions(junctions)

    new_tcrs = resample_cdr3_nt_regions(junctions)

    exit()


if 0: # look at YFV nndists for expanding clones
    # this made the colorful NNbr-distances figure
    import faiss

    runtag = 'run14' # d0 10 nbrs, bgnums 4,5
    #runtag = 'run12' # d15 25 nbrs
    #runtag = 'run11' # d15 10 nbrs
    bgnum = 4
    aa_mds_dim = 8 # for finding nbrs of expanding clones

    runprefix = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/{runtag}'

    # read expanding clones
    xclones = pd.read_table(
        '/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/yfv_expanded_clones.tsv')
    xclones.rename(columns={
        'CDR3.nucleotide.sequence':'cdr3nt',
        'bestVGene':'v',
        'bestJGene':'j',
        'CDR3.amino.acid.sequence':'cdr3aa',
        }, inplace=True)
    xclones['v'] = xclones.v+'*01'
    xclones['j'] = xclones.j+'*01'
    xclones['cdr3nt'] = xclones.cdr3nt.str.lower()
    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    xclones = filter_out_bad_genes_and_cdr3s(
        xclones, v_column, cdr3_column, organism, chain, j_column=j_column)
    xclones['aatcr'] = xclones.v + '_' + xclones.cdr3aa
    print('num_xclones:', xclones.shape[0])

    if runtag == 'run14':
        fg_files = sorted(glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/'
                               'Yellow_fever/??_0_*min_count_2.txt.gz'))
        assert len(fg_files) == 12
        nrows, ncols = 3, 4
    else:
        fg_files=sorted(glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/'
                             'Yellow_fever/??_15_*min_count_2.txt.gz'))
        assert len(fg_files) == 7
        nrows, ncols = 2, 4
    #fg_files = fg_files[2:3]

    plt.figure(figsize=(ncols*4, nrows*4))

    for plotno, fg_file in enumerate(fg_files):
        fg_tag = fg_file.split('/')[-1][:-3]
        donor = fg_tag[:2]

        tcrs = read_britanova_tcrs(fg_file)
        tcrs['aatcr'] = tcrs.v + '_' + tcrs.cdr3aa
        num_tcrs = tcrs.shape[0]

        vecs = gapped_encode_tcr_chains(
            tcrs, organism, chain, aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column).astype(np.float32)

        xvecs = gapped_encode_tcr_chains(
            xclones[xclones.donor==donor], organism, chain, aa_mds_dim,
            v_column=v_column, cdr3_column=cdr3_column).astype(np.float32)


        idx = faiss.IndexFlatL2(vecs.shape[1])
        idx.add(vecs)
        start = timer()
        print('run range search')
        lims,D,I = idx.range_search(xvecs, 12.5)
        print(f'xvecs range_search took {timer()-start:.2f} seconds')
        xnbrs1_mask = np.zeros((num_tcrs,)).astype(bool)
        xnbrs1_mask[I] = True

        start = timer()
        print('run range search')
        lims,D,I = idx.range_search(xvecs, 24.5)
        print(f'xvecs range_search took {timer()-start:.2f} seconds')
        xnbrs2_mask = np.zeros((num_tcrs,)).astype(bool)
        xnbrs2_mask[I] = True


        xtcrs = set(xclones[xclones.donor==donor].aatcr)
        xmask = tcrs.aatcr.isin(xtcrs)
        print('num xtcrs:', xmask.sum(), len(set(tcrs[xmask].aatcr)), len(xtcrs))

        fname = f'{runprefix}_{fg_tag}_r0_fg_nndists.npy'
        print(fname)
        assert exists(fname)

        fg_nndists = np.load(fname)
        assert fg_nndists.shape == (num_tcrs,)

        bg_nndists = np.zeros((num_tcrs,))
        for r in range(10):
            fname = f'{runprefix}_{fg_tag}_r0_bg_{bgnum}_nndists.npy'
            bg_nndists += np.load(fname)
        bg_nndists /= 10

        plt.subplot(nrows, ncols, plotno+1)
        plt.scatter(fg_nndists, bg_nndists, s=5, alpha=0.2)
        plt.plot([0,100],[0,100],':k')
        plt.title(fg_tag)

        plt.scatter(fg_nndists[xnbrs2_mask], bg_nndists[xnbrs2_mask], s=5, alpha=0.5,
                    label='dist<24.5 to expd. clone')
        plt.scatter(fg_nndists[xnbrs1_mask], bg_nndists[xnbrs1_mask], s=5, alpha=0.5,
                    label='dist<12.5 to expd. clone')

        xinds = np.nonzero(np.array(xmask))[0]
        plt.scatter(fg_nndists[xinds], bg_nndists[xinds], s=5, alpha=1,
                    label='YFV-expanding clone')
        plt.legend(fontsize=8, loc='lower right')
        if plotno%ncols==0:
            plt.ylabel('avg dist to 10 NNbrs (10 background reps)')
        if plotno//ncols==nrows-1:
            plt.xlabel('avg dist to 10 NNbrs')



    plt.tight_layout()

    pngfile = ('/home/pbradley/csdat/raptcr/'
               f'yfv_{runtag}_nndists_F{len(fg_files)}_bg{bgnum}.png')
    plt.savefig(pngfile, dpi=300)
    print('made:', pngfile)

    exit()

if 0: # look at nbr counts in YFV data
    # this made the figure with the Evalue panels and the tcr-graph panels
    #
    import tcrdist
    import networkx as nx
    from scipy.spatial.distance import squareform, pdist
    import faiss

    aa_mds_dim = 8
    radius = 12.5
    bgnum = 4
    max_evalue = 0.1
    num_lines = 50
    target_bg_nbrs = None
    COLOR_GRAPH_BY_EVALUE = False

    # fg_files=sorted(glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
    #                        '??_0_*min_count_2.txt.gz'))
    # assert len(fg_files) == 12
    # prefix_for_bg = '/home/pbradley/csdat/raptcr/slurm/run13/run13'
    # prefix_for_cb = '/home/pbradley/csdat/raptcr/slurm/run15/run15'
    # bg_nums = [4,5]
    fg_files= sorted(glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
                            '??_15_*min_count_2.txt.gz'))
    assert len(fg_files) == 7
    prefix_for_bg = '/home/pbradley/csdat/raptcr/slurm/run9/run9' # day 15
    prefix_for_cb = '/home/pbradley/csdat/raptcr/slurm/run10/run10'
    bg_nums = range(6)


    #fg_files = fg_files[3:4]

    #fg_files = fg_files[-2:]
    #fg_files = fg_files[:1]
    #fg_files = fg_files[:5]

    # read expanding clones
    xclones = pd.read_table(
        '/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/yfv_expanded_clones.tsv')
    xclones.rename(columns={
        'CDR3.nucleotide.sequence':'cdr3nt',
        'bestVGene':'v',
        'bestJGene':'j',
        'CDR3.amino.acid.sequence':'cdr3aa',
        }, inplace=True)
    xclones['v'] = xclones.v+'*01'
    xclones['j'] = xclones.j+'*01'
    xclones['cdr3nt'] = xclones.cdr3nt.str.lower()
    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    xclones = filter_out_bad_genes_and_cdr3s(
        xclones, v_column, cdr3_column, organism, chain, j_column=j_column)
    print('num_xclones:', xclones.shape[0])
    tcrdister = tcrdist.tcr_distances.TcrDistCalculator('human')
    # aa_mds_dim = 8
    # xvecs = gapped_encode_tcr_chains(
    #     xclones, organism, chain, aa_mds_dim, v_column=v_column,
    #     cdr3_column=cdr3_column).astype(np.float32)

    #exit()



    plotno_scale = 2+COLOR_GRAPH_BY_EVALUE
    nplots = plotno_scale*len(fg_files)
    nrows = max(1, int(0.8*np.sqrt(nplots)))
    ncols = (nplots-1)//nrows + 1
    print(nrows, ncols, fg_files)
    nrows, ncols = 2,7 # hack
    plt.figure(figsize=(ncols*3, nrows*3))


    pngfile = ('/home/pbradley/csdat/raptcr/yfv_'+prefix_for_bg.split('/')[-1]+
               f'_pvals_F{len(fg_files)}_{radius:.1f}_bg{bgnum}_'
               f'tbn{target_bg_nbrs}.png')

    # for getting the sizes of the britanova cord blood reps
    run4prefix = '/home/pbradley/csdat/raptcr/slurm/run4/run4'

    # read the rep sizes
    print('reading all rep sizes')
    files = glob(f'{run4prefix}_A*_{radius:.1f}_r0_fg_nbr_counts.npy')
    all_rep_sizes = {}
    for fname in files:
        ftag = fname.split('_')[-6]
        counts = np.load(fname)
        all_rep_sizes[ftag] = counts.shape[0]
    print('DONE reading all rep sizes')

    for plotno, fg_file in enumerate(fg_files):
        fg_tag = fg_file.split('/')[-1][:-3]
        tcrs = read_britanova_tcrs(fg_file)

        my_counts_files = glob(f'{prefix_for_bg}_{fg_tag}_{radius:.1f}_*npy')
        print(fg_tag, 'numfiles:', len(my_counts_files))
        # if len(my_counts_files)<70:
        #     continue
        # assert len(my_counts_files) == 70

        # read fg counts
        fg_counts=np.load(f'{prefix_for_bg}_{fg_tag}_{radius:.1f}_r0_fg_nbr_counts.npy')
        num_tcrs = fg_counts.shape[0]
        assert num_tcrs == tcrs.shape[0]

        # read bg counts, compute pvals
        all_bg_counts = {}
        for bg in bg_nums:
            print('reading bg counts:', bg)
            bg_counts = np.zeros((num_tcrs,))
            for r in range(10):
                bg_counts += np.load(f'{prefix_for_bg}_{fg_tag}_{radius:.1f}_r{r}_bg_'
                                     f'{bg}_nbr_counts.npy')
            all_bg_counts[bg] = bg_counts

        bg_counts = all_bg_counts[bgnum]
        num_bg_tcrs = 10*num_tcrs


        min_fg_bg_nbr_ratio = 2. # actually these are the function defaults
        max_fg_bg_nbr_ratio = 100
        pvals = compute_nbr_count_pvalues(
            fg_counts, bg_counts, num_bg_tcrs,
            min_fg_bg_nbr_ratio=min_fg_bg_nbr_ratio,
            max_fg_bg_nbr_ratio=max_fg_bg_nbr_ratio,
            target_bg_nbrs=target_bg_nbrs,
        )
        #if 2:
        #    print('using rescaled evalues!')
        #    pvals['evalue'] = pvals.rescaled_evalue
        pvals = pvals.join(tcrs['count v j cdr3nt cdr3aa'.split()].reset_index(),
                           on='tcr_index')
        pvals = pvals.sort_values('evalue').drop_duplicates(['v','j','cdr3aa'])
        pvals = pvals[pvals.evalue<= max_evalue]


        donor = fg_tag[:2] ; assert donor in 'P1 P2 Q1 Q2 S1 S2'.split()
        xclone_dists = calc_tcrdist_matrix(pvals, xclones[xclones.donor==donor],
                                           tcrdister)

        #read nbr counts in other britanova reps
        fgfiles = glob(f'{prefix_for_cb}_{fg_tag}_{radius:.1f}_'
                       'bg_A5-*.txt_nbr_counts.npy')

        all_fg_counts = {}
        for fname in fgfiles:
            other_ftag = fname.split('_')[-3]
            all_fg_counts[other_ftag] = np.load(fname)


        print('='*80)
        for ii, l in enumerate(pvals.head(num_lines).itertuples()):
            ind = l.tcr_index
            #tcr = tcrs.iloc[ind]
            mindist = int(min(xclone_dists[ii,:]))
            same_tcr_mask = (tcrs.v==l.v)&(tcrs.j==l.j)&(tcrs.cdr3aa==l.cdr3aa)
            num_clones = same_tcr_mask.sum()
            num_cells = sum(tcrs[same_tcr_mask]['count'])
            obs = fg_counts[ind]
            msg= f'eval: {l.evalue:9.2e} {obs:3d} {l.expected_nbrs:5.1f} bg%'
            for bg in bg_nums:
                expect = all_bg_counts[bg][ind] / 10.
                msg += f' {100*expect/obs:3.0f}'
            msg += ' fg%'
            for other_ftag in sorted(all_fg_counts.keys()):
                expect = ((all_fg_counts[other_ftag][ind]-(fg_tag==other_ftag)) *
                          num_tcrs / all_rep_sizes[other_ftag])
                msg += f' {100*expect/obs:3.0f}'
            msg += (f' {fg_tag} {mindist:3d} {num_clones:2d} {num_cells:4d} '
                    f'{l.v} {l.j} {l.cdr3aa}')
            print(msg)



        # plotting ###
        #plt.subplot(nrows, ncols, plotno_scale*plotno+1)
        plt.subplot(2,7,plotno+1) # hack
        plt.title(f'{fg_tag[:9]} N={num_tcrs} R={radius:.1f} {pvals.shape[0]}',
                  fontsize=7)
        #bg_scale = num_bg_tcrs/num_tcrs
        xvals = np.log10(1+pvals.expected_nbrs)
        yvals = -1*np.log10(pvals.evalue)
        cvals = np.log10(pvals.fg_bg_nbr_ratio)

        plt.scatter(xvals, yvals, c=cvals,
                    vmin=np.log10(min_fg_bg_nbr_ratio),
                    vmax=np.log10(max_fg_bg_nbr_ratio))
        #plt.xlim((-0.025, plt.xlim()[1]))
        locs,_ = plt.xticks()
        labs = [f'{10**x - 1:.1f}' for x in locs]
        mn,mx = plt.xlim()
        plt.xticks(locs,labs)
        plt.xlim((mn,mx)) # dunno why we need this?!?
        mn,mx = plt.ylim()
        plt.ylim((-0.1, max(mx,15.)))
        plt.xlabel('expected_num_nbrs (based on sim. background)', fontsize=6)
        if plotno==0:
            plt.ylabel('-1*log_10(E-value)')


        ## draw a graph of the significant tcrs with edges between tcrs whose
        ## distance is less than radius
        goodvecs = gapped_encode_tcr_chains(
            pvals, organism, chain, aa_mds_dim=aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column).astype(np.float32)
        idx = faiss.IndexFlatL2(goodvecs.shape[1])
        idx.add(goodvecs)
        start = timer()
        lims,D,I = idx.range_search(goodvecs, radius)

        dists = squareform(pdist(goodvecs))**2

        nbrlist = list(zip(*np.nonzero(dists<=radius)))
        print('nbrs:', len(nbrlist), len(D))
        g = nx.Graph()
        for i in range(pvals.shape[0]):
            g.add_node(i)
        for i,j in nbrlist:
            if i<j:
                g.add_edge(i,j)

        # right now these labels are not being used...
        comps = list(nx.connected_components(g))
        print('num comps:', len(comps))
        labels = {x:'' for x in range(pvals.shape[0])}

        for comp in comps:
            evalue, irep = min((pvals.iloc[x]['evalue'],x) for x in comp)
            #print(len(comp), nndist, rep)
            rep = pvals.iloc[irep]
            labels[irep] = f'{rep.v} {rep.cdr3aa}'


        k = 2/np.sqrt(pvals.shape[0]) # default is 1/sqrt(N)
        pos = nx.drawing.layout.spring_layout(g, k=k)

        if COLOR_GRAPH_BY_EVALUE:
            plt.subplot(nrows, ncols, plotno_scale*plotno+2)
            plt.title('color by evalue')
            colors = [-1*np.log10(pvals.iloc[x]['evalue']) for x in list(g)]
            nx.draw_networkx(g, pos, ax=plt.gca(), node_size=10, with_labels=False,
                             # labels=labels,
                             node_color=colors)#, vmin=0, vmax = 48)

        #plt.subplot(nrows, ncols, plotno_scale*plotno+2+COLOR_GRAPH_BY_EVALUE)
        plt.subplot(2,7,7+plotno+1) # hack
        plt.title('tcr graph colored by mindist to YFV-expanding clone\n'
                  '(blue: tcrdist=0 to yellow: tcrdist>=48', fontsize=8)
        min_xclonedists = xclone_dists.min(axis=1)
        assert min_xclonedists.shape == (pvals.shape[0],)
        colors = [min_xclonedists[x] for x in list(g)]
        nx.draw_networkx(g, pos, ax=plt.gca(), node_size=10, with_labels=False,
                         # labels=labels,
                         node_color=colors, vmin=0, vmax = 48)
        plt.ylabel('nodes are tcrs w/ significant E-values', fontsize=8)
        plt.xlabel(f'edges connect tcrs w/ dist<= {radius:.1f}', fontsize=8)


        #run5_A3-i107.txt_6.5_bg_A5-S11.txt_nbr_counts.npy
    plt.tight_layout()
    plt.savefig(pngfile, dpi=100)
    print('made:', pngfile)

    exit()


if 0: # look at nbr counts in the britanova set
    #old_ftags = get_original_18_ftags()
    #print(old_ftags)
    #exit()

    britdir = DATADIR+'phil/britanova/'
    metadata = pd.read_table(britdir+'metadata.txt')

    fg_files = sorted(glob(britdir+'A*.txt.gz'))
    assert len(fg_files) == metadata.shape[0]

    fg_files = sorted(
        [britdir+x.file_name+'.gz' for x in metadata.itertuples() if x.age==0])

    fg_files = get_original_18_fnames()

    #fg_files = fg_files[-1:]

    #fg_files = fg_files[-2:]
    #fg_files = fg_files[:1]
    #fg_files = fg_files[:5]

    radius = 12.5
    bgnum = 7
    max_evalue = 10 #0.1
    #max_evalue = 0.1
    num_lines = 50
    target_bg_nbrs = None

    nrows = max(1, int(0.75*np.sqrt(len(fg_files))))
    ncols = (len(fg_files)-1)//nrows + 1
    print(nrows, ncols, fg_files)
    plt.figure(figsize=(ncols*3, nrows*3))
    pngfile = ('/home/pbradley/csdat/raptcr/'
               f'run4run5_pvals_F{len(fg_files)}_{radius:.1f}_bg{bgnum}_'
               f'tbn{target_bg_nbrs}.png')

    # this is very silly:
    run4prefix = '/home/pbradley/csdat/raptcr/slurm/run4/run4' # for fg vs fg
    run5prefix = '/home/pbradley/csdat/raptcr/slurm/run5/run5' # fg vs cord blood
    bg_runtags = {0:4, 1:4, 2:4, 3:4, 4:4, 5:8, 6:16, 7:19}

    # read the rep sizes
    print('reading all rep sizes')
    files = glob(f'{run4prefix}_A*_{radius:.1f}_r0_fg_nbr_counts.npy')
    all_rep_sizes = {}
    for fname in files:
        ftag = fname.split('_')[-6]
        counts = np.load(fname)
        all_rep_sizes[ftag] = counts.shape[0]
    print('DONE reading all rep sizes')

    for plotno, fg_file in enumerate(fg_files):
        fg_tag = fg_file.split('/')[-1][:-3]
        tcrs = read_britanova_tcrs(fg_file)

        # read fg counts
        fg_counts = np.load(f'{run4prefix}_{fg_tag}_{radius:.1f}_r0_fg_nbr_counts.npy')
        num_tcrs = fg_counts.shape[0]
        assert num_tcrs == tcrs.shape[0]
        assert num_tcrs == all_rep_sizes[fg_tag]

        # read bg counts, compute pvals
        all_bg_counts = {}
        for bg in range(8):
            print('reading bg counts:', bg, flush=True)
            bg_counts = np.zeros((num_tcrs,))
            for r in range(10):
                rt = 'run'+str(bg_runtags[bg])
                prefix = f'/home/pbradley/csdat/raptcr/slurm/{rt}/{rt}'
                bg_counts += np.load(f'{prefix}_{fg_tag}_{radius:.1f}_r{r}_bg_'
                                     f'{bg}_nbr_counts.npy')
            all_bg_counts[bg] = bg_counts

        bg_counts = all_bg_counts[bgnum]
        num_bg_tcrs = 10*num_tcrs


        min_fg_bg_nbr_ratio = 2. # actually these are the function defaults
        max_fg_bg_nbr_ratio = 100
        pvals = compute_nbr_count_pvalues(
            fg_counts, bg_counts, num_bg_tcrs,
            min_fg_bg_nbr_ratio=min_fg_bg_nbr_ratio,
            max_fg_bg_nbr_ratio=max_fg_bg_nbr_ratio,
            target_bg_nbrs=target_bg_nbrs,
        )
        if 2 and target_bg_nbrs is not None:
            print('using rescaled evalues!')
            pvals['evalue'] = pvals.rescaled_evalue
        pvals = pvals.join(tcrs['count v j cdr3nt cdr3aa'.split()].reset_index(),
                           on='tcr_index')
        pvals = pvals.sort_values('evalue').drop_duplicates(['v','j','cdr3aa'])
        pvals = pvals[pvals.evalue<= max_evalue]

        # read nbr counts in other britanova reps
        fgfiles = glob(f'{run5prefix}_{fg_tag}_{radius:.1f}_'
                       'bg_A5-*.txt_nbr_counts.npy')

        all_fg_counts = {}
        for fname in fgfiles:
            other_ftag = fname.split('_')[-3]
            all_fg_counts[other_ftag] = np.load(fname)


        print('='*80)
        for l in pvals.head(num_lines).itertuples():
            ind = l.tcr_index
            #tcr = tcrs.iloc[ind]
            same_tcr_mask = (tcrs.v==l.v)&(tcrs.j==l.j)&(tcrs.cdr3aa==l.cdr3aa)
            num_clones = same_tcr_mask.sum()
            num_cells = sum(tcrs[same_tcr_mask]['count'])
            obs = fg_counts[ind]
            msg= f'eval: {l.evalue:9.2e} {obs:3d} {l.expected_nbrs:5.1f} bg% '
            for bg in range(8):
                expect = all_bg_counts[bg][ind] / 10.
                msg += f' {100*expect/obs:3.0f}'
            msg += '  fg% '
            for other_ftag in sorted(all_fg_counts.keys()):
                #if other_ftag == fg_tag:
                #    continue
                expect = ((all_fg_counts[other_ftag][ind]-(fg_tag==other_ftag)) *
                          all_rep_sizes[fg_tag] / all_rep_sizes[other_ftag])
                msg += f' {100*expect/obs:3.0f}'
            msg += f'  {fg_tag} {num_clones:2d} {num_cells:4d} {l.v} {l.j} {l.cdr3aa}'
            print(msg)



        # plotting ###
        plt.subplot(nrows, ncols, plotno+1)
        plt.title(f'{fg_tag} N={num_tcrs} R={radius:.1f} {pvals.shape[0]}',
                  fontsize=8)
        #bg_scale = num_bg_tcrs/num_tcrs
        xvals = np.log10(1+pvals.expected_nbrs)
        yvals = -1*np.log10(pvals.evalue)
        cvals = np.log10(pvals.fg_bg_nbr_ratio)

        plt.scatter(xvals, yvals, c=cvals,
                    vmin=np.log10(min_fg_bg_nbr_ratio),
                    vmax=np.log10(max_fg_bg_nbr_ratio))
        mn,mx = plt.ylim()
        plt.ylim((-0.1, max(mx,15.)))



        #run5_A3-i107.txt_6.5_bg_A5-S11.txt_nbr_counts.npy
    plt.tight_layout()
    plt.savefig(pngfile, dpi=100)
    print('made:', pngfile)

    exit()



if 0: # make multi-panel plot of pvals across different radii and britanova
    # repertoires
    from scipy.stats import hypergeom

    runtag = 'run4' ; max_tcrs = 500000
    #runtag = 'run1' ; max_tcrs = 100000
    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    files = glob(f'{rundir}{runtag}*nbr_counts.npy')

    ftags = sorted(set(x.split('/')[-1].split('_')[1] for x in files))
    radii = sorted(set(float(x.split('/')[-1].split('_')[2]) for x in files))

    min_fg_nbrs = 2
    max_expected_fg_nbrs = 500000 #5

    bgnums = [4]
    #bgnums = list(range(5))

    my_ftags = ftags[:]
    #my_ftags = ftags[:1]+ftags[-1:]
    #my_ftags = ftags[:5]+ftags[-5:]
    #my_ftags = ftags[:1]

    nrows = len(my_ftags)
    ncols = len(radii)
    plt.figure(figsize=(ncols*4, nrows*4))

    for row, ftag in enumerate(my_ftags):
        fname = ('/home/pbradley/gitrepos/immune_response_detection/data/phil/'
                 f'britanova/{ftag}.gz')
        tcrs = read_britanova_tcrs(fname, max_tcrs=max_tcrs)
        num_tcrs = tcrs.shape[0]
        print('num_tcrs:', num_tcrs, ftag)

        for col, radius in enumerate(radii):
            fname = f'{rundir}{runtag}_{ftag}_{radius:.1f}_r0_fg_nbr_counts.npy'
            fg_counts = np.load(fname)
            assert num_tcrs == fg_counts.shape[0]
            bg_counts = np.zeros((num_tcrs,))
            num_bg_reps = 0
            for r in range(10):
                for bgnum in bgnums:
                    fname = (f'{rundir}{runtag}_{ftag}_{radius:.1f}_r{r}_bg_{bgnum}_'
                             'nbr_counts.npy')
                    counts = np.load(fname)
                    assert counts.shape[0] == num_tcrs
                    bg_counts += counts
                    num_bg_reps += 1
            num_bg_tcrs = num_tcrs * num_bg_reps

            min_fg_bg_nbr_ratio = 2. # actually these are the function defaults
            max_fg_bg_nbr_ratio = 100
            pvals = compute_nbr_count_pvalues(
                fg_counts, bg_counts, num_bg_tcrs,
                min_fg_bg_nbr_ratio=min_fg_bg_nbr_ratio,
                max_fg_bg_nbr_ratio=max_fg_bg_nbr_ratio,
                target_bg_nbrs=None,
            )

            mask = (pvals.evalue<0.05)
            print('num_sig:', mask.sum(), radius, ftag)
            pvals = pvals[mask]
            plt.subplot(nrows, ncols, row*ncols+col+1)
            plt.title(f'{ftag} N={num_tcrs} R={radius:.1f} {mask.sum()}')
            #bg_scale = num_bg_tcrs/num_tcrs
            xvals = np.log10(1+pvals.expected_nbrs)
            yvals = -1*np.log10(pvals.evalue)
            cvals = np.log10(pvals.fg_bg_nbr_ratio)
            plt.scatter(xvals, yvals, c=cvals,
                        vmin=np.log10(min_fg_bg_nbr_ratio),
                        vmax=np.log10(max_fg_bg_nbr_ratio))

            if row==nrows-1:
                plt.xlabel('log_10(1+expected_num_nbrs)')
            if col==0:
                plt.ylabel('-1*log_10(E-value)')
            if col==ncols-1:
                plt.colorbar()
    plt.tight_layout()
    bgtag = ''.join(map(str,bgnums))
    pngfile = f'/home/pbradley/csdat/raptcr/{runtag}_brit_pvals_bg{bgtag}.png'
    plt.savefig(pngfile, dpi=100)
    print('made:', pngfile)
    exit()




if 0: # setup for big cluster calc, brit-vs-brit nbrs
    # this block generates a file containing commands
    # those commands get distributed over the cluster
    #
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    radii = [12.5, 18.5, 24.5]
    #radii = [3.5, 6.5, 12.5, 18.5, 24.5]

    britdir = DATADIR+'phil/britanova/'
    yfvdir = '/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
    df = pd.read_table(britdir+'metadata.txt')


    fg_fnames = sorted(glob(yfvdir+'??_0_*count_2.txt.gz')) ; assert len(fg_fnames)==12
    #fg_fnames = sorted(glob(yfvdir+'*count_2.txt.gz')) ; assert len(fg_fnames)==7
    #fg_fnames = [f'{britdir}{x}.gz' for x in df.file_name]
    bg_fnames = [f'{britdir}{x}.gz' for x,y in zip(df.file_name, df.age) if y==0]

    # fnames = glob('/home/pbradley/gitrepos/immune_response_detection/'
    #               'data/phil/britanova/A*gz')
    print(len(fg_fnames), len(bg_fnames))

    runtag = 'run15' ; xargs = ' --max_tcrs 500000 ' # now yfv day0
    #runtag = 'run10' ; xargs = ' --max_tcrs 500000 ' # now yfv day15
    #runtag = 'run7' ; xargs = ' --max_tcrs 500000 ' # now full britanova download
    #runtag = 'run5' ; xargs = ' --max_tcrs 500000 ' # original set of 18

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    for fname in fg_fnames:
        ftag = fname.split('/')[-1][:-3]
        for radius in radii:
            for bg_fname in bg_fnames:
                outfile_prefix = f'{rundir}{runtag}_{ftag}_{radius:.1f}'
                cmd = (f'{PY} {EXE} {xargs} --mode brit_vs_brit '
                       f' --fg_filename {fname} --radius {radius} '
                       f' --bg_filenames {bg_fname} '
                       f' --outfile_prefix {outfile_prefix} '
                       f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
                out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)

    exit()

if 0: # setup for big nndists calc on cluster
    # this block generates a file containing commands
    # those commands get distributed over the cluster
    #
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    num_repeats = 10

    fnames = glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
                  '??_0_*min_count_2.txt.gz') ; assert len(fnames) == 12
    # fnames = glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
    #               '*min_count_2.txt.gz')
    #fnames = glob('/home/pbradley/gitrepos/immune_response_detection/'
    #              'data/phil/britanova/A*gz')
    print(len(fnames))


    runtag = 'run14' ; xargs = ' --num_nbrs 10 '; bg_nums = [4,5]
    # runtag = 'run12' ; xargs = ' --num_nbrs 25 '
    #runtag = 'run11' ; xargs = ' --num_nbrs 10 '

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    for fname in fnames:
        for bgnum in bg_nums:
            ftag = fname.split('/')[-1][:-3]
            for repeat in range(num_repeats):
                xxargs = ' --skip_fg ' if repeat>0 else ''
                outfile_prefix = f'{rundir}{runtag}_{ftag}_r{repeat}'
                cmd = (f'{PY} {EXE} {xargs} {xxargs} --mode nndists_vs_bg '
                       f' --bg_nums {bgnum} '
                       f' --filename {fname} --outfile_prefix {outfile_prefix} '
                       f' > {outfile_prefix}_{bgnum}.log '
                       f' 2> {outfile_prefix}_{bgnum}.err')
                out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)

    exit()



if 0: # setup for big calc
    # this block generates a file containing commands
    # those commands get distributed over the cluster
    #
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    radii = [12.5, 18.5, 24.5]
    #radii = [3.5, 6.5, 12.5, 18.5, 24.5]
    num_repeats = 10
    bg_nums = [7]
    #bg_nums = [4,5]

    # fnames = glob('/home/pbradley/csdat/raptcr/emerson/*reparsed.tsv')
    # assert len(fnames) == 666
    # old_fnames = set(get_emerson_files_for_allele('A*02:01'))
    # fnames = [x for x in fnames if x not in old_fnames]
    # assert len(fnames) == 666 - 268
    # fnames = get_emerson_files_for_allele('A*02:01')

    # fnames = glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
    #               '??_0_*min_count_2.txt.gz') ; assert len(fnames) == 12
    fnames = glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
                  '??_15_*min_count_2.txt.gz') ; assert len(fnames) == 7
    # fnames = glob('/home/pbradley/gitrepos/immune_response_detection/'
    #               'data/phil/britanova/A*gz')
    print(len(fnames))

    runtag = 'run9c' ; xargs = ' --max_tcrs 500000 ' # yfv day 15, bg_nums = [6]
    # runtag = 'run9b' ; xargs = ' --max_tcrs 500000 ' # yfv day 15, bg_nums = [6]
    #runtag = 'run19' ; xargs = ' --max_tcrs 500000 ' # britanova, bg_nums=[7]
    #runtag = 'run18' ; xargs = ' --max_tcrs 500000 ' # emerson, A*02:01-/unk donors
    #runtag = 'run17' ; xargs = ' --max_tcrs 500000 ' # emerson, A*02:01+ donors
    #runtag = 'run16' ; xargs = ' --max_tcrs 500000 ' # britanova, bg_nums=[4,6]
    #runtag = 'run13' ; xargs = ' --max_tcrs 500000 ' # yfv day 0
    #runtag = 'run9' ; xargs = ' --max_tcrs 500000 ' # yfv day 15
    #runtag = 'run8' ; xargs = ' --max_tcrs 500000 --bg_nums 5 ' # new bg model
    #runtag = 'run6' ; xargs = ' --max_tcrs 500000 ' # new repertoires
    #runtag = 'run4' ; xargs = ' --max_tcrs 500000 ' # new 5th bg rep
    #runtag = 'run3' ; xargs = ' --max_tcrs 500000 '
    #runtag = 'run2' ; xargs = ' --aa_mds_dim 16 --max_tcrs 100000 '
    #runtag = 'run1' ; xargs = ' --max_tcrs 100000 '

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    for fname in fnames:
        ftag = fname.split('/')[-1][:-3]
        for radius in radii:
            for repeat in range(num_repeats):
                for bg_num in bg_nums:
                    xxargs = ' --skip_fg ' if repeat>0 else ''
                    outfile_prefix = f'{rundir}{runtag}_{ftag}_{radius:.1f}_r{repeat}'
                    cmd = (f'{PY} {EXE} {xargs} {xxargs} --mode brit_vs_bg '
                           f' --filename {fname} --radius {radius} --bg_nums {bg_num} '
                           f' --outfile_prefix {outfile_prefix} '
                           f' > {outfile_prefix}_{bg_num}.log '
                           f' 2> {outfile_prefix}_{bg_num}.err')
                    out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)

    exit()





if 0: # explore a super-simple background repertoire
    from pynndescent import NNDescent

    aa_mds_dim = 8 # per-aa MDS embedding dimension

    organism, chain, v_column, cdr3_column = 'human', 'B','v_call', 'junction_aa'
    j_column = 'j_call'
    df = pd.read_table('./data/example_repertoire.tsv')#.head(10000)

    df = filter_out_bad_genes_and_cdr3s(
        df, v_column, cdr3_column, organism, chain, j_column=j_column)

    parsed_df = parse_repertoire(df, organism, chain, v_column, j_column, cdr3_column)

    bg_df = resample_parsed_repertoire(parsed_df, df.shape[0])
    bg_df.rename(columns={'v':v_column, 'cdr3':cdr3_column}, inplace=True)
    fg_df = df

    for tag, df in zip(['fg','bg'], [fg_df, bg_df]):

        start = timer()
        vecs = gapped_encode_tcr_chains(
            df, organism, chain, aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column)
        print(f'gapped_encode_tcr_chains: {aa_mds_dim} {timer()-start:.6f}')

        print('training the index...', vecs.shape)
        start = timer()
        index = NNDescent(
            vecs, n_neighbors=10, diversify_prob=1.0, pruning_degree_multiplier=1.5,
        )
        print(f'training took {timer()-start:.3f}')

        I,D = index.neighbor_graph

        nndists = np.mean(D**2, axis=-1)

        # show tcrs with smallest nndist
        top_inds = np.argsort(nndists)[:50]

        for ii, ind in enumerate(top_inds):
            print(f'{tag} {ii:3d} {nndists[ind]:7.2f} {ind:6d}',
                  df.iloc[ind].tolist())

if 0: # find v+cdr3 tcrdist nbrs in tcrb example dataset using naive gapped encoding
    # playing around with pynndescent index
    #
    from tcrdist.all_genes import all_genes
    from pynndescent import NNDescent

    organism, chain, v_column, cdr3_column = 'human', 'B','v_call', 'junction_aa'

    df = pd.read_table('./data/example_repertoire.tsv')#.head(10000)

    df = filter_out_bad_genes_and_cdr3s(df, v_column, cdr3_column, organism, chain)

    aa_mds_dim = 8 # per-aa MDS embedding dimension

    start = timer()
    vecs = gapped_encode_tcr_chains(
        df, organism, chain, aa_mds_dim, v_column=v_column,
        cdr3_column=cdr3_column)
    print(f'gapped_encode_tcr_chains: {aa_mds_dim} {timer()-start:.6f}')

    print('training the index...', vecs.shape)
    start = timer()
    index = NNDescent(
        vecs, n_neighbors=10, diversify_prob=1.0, pruning_degree_multiplier=1.5,
    )
    print(f'training took {timer()-start:.3f}')
    if 2: # dunno why it's faster the second time... maybe some import statements?
        start = timer()
        index = NNDescent(
            vecs, n_neighbors=10, diversify_prob=1.0, pruning_degree_multiplier=1.5,
        )
        print(f'training AGAIN took {timer()-start:.3f}')

    I,D = index.neighbor_graph

    nndists = np.mean(D, axis=-1)

    # show tcrs with smallest nndist
    top_inds = np.argsort(nndists)[:100]

    for ind in top_inds:
        print(ind, nndists[ind], df.iloc[ind].tolist())



if 0: # try encoding full tcr chains
    #fname = './data/phil/conga_lit_db.tsv'
    from scipy.spatial.distance import pdist, squareform
    import tcrdist
    from tcrdist.tcr_distances import TcrDistCalculator
    from scipy.stats import linregress
    organism = 'human'

    fname = './data/phil/pregibon_tcrs.tsv'
    df = pd.read_table(fname)

    ms = [4,8,12,16,32,64]

    tcrdister = TcrDistCalculator(organism)

    nrows, ncols = 2, len(ms)
    plt.figure(figsize=(4*ncols, 4*nrows))

    for row, chain in enumerate('AB'):
        v_column, cdr3_column = 'v'+chain.lower(), 'cdr3'+chain.lower()
        tcrs = [(x[v_column],'',x[cdr3_column]) for _,x in df.iterrows()]
        N = df.shape[0]
        start = timer()
        tcrD = np.array([tcrdister.single_chain_distance(x,y)
                         for x in tcrs for y in tcrs]).reshape(N,N)
        print(f'tcrdist calc: {timer()-start:.6f}')

        for col,m in enumerate(ms):
            plt.subplot(nrows, ncols, row*ncols+col+1)
            start = timer()
            vecs = gapped_encode_tcr_chains(
                df, organism, chain, m, v_column=v_column,
                cdr3_column=cdr3_column)
            print(f'gapped_encode_tcr_chains: {m} {timer()-start:.6f}')

            start = timer()
            rapD = squareform(pdist(vecs))**2
            print(f'pdist calc: {m} {timer()-start:.6f}')

            reg = linregress(rapD.ravel(), tcrD.ravel())
            plt.title(f'chain= {chain} aa_mds_dim= {m}\nveclen= {vecs.shape[1]} '
                      f'Rvalue= {reg.rvalue:.6f}')
            plt.scatter(rapD.ravel(), tcrD.ravel(), s=5)
            plt.xlabel('encoded pdists')
            plt.ylabel('tcrdists')
    plt.tight_layout()
    pngfile = f'vcdr3dist_comparisons_{df.shape[0]}.png'
    pngfile = '/home/pbradley/csdat/raptcr/'+pngfile
    plt.savefig(pngfile)
    print('made:', pngfile)
    #plt.show()
    plt.close('all')





if 0: # look at cdr lengths
    from collections import Counter
    from tcrdist.all_genes import all_genes

    organism = 'human'
    for ab in 'AB':

        cdrs = setup_gene_cdr_strings(organism, ab)
        vals = list(cdrs.values())
        assert all(len(x) == len(vals[0]) for x in vals)
        print('\n'.join(f'{len(x)} {x}' for x in vals[:10]))
        continue

        for icdr in range(3):
            genes = [x for x in all_genes[organism] if x[2:4] == ab+'V']
            cdrs = [all_genes[organism][x].cdrs[icdr] for x in genes]
            lens = [len(x) - x.count('.') for x in cdrs]
            print(ab, icdr, sorted(Counter(lens).most_common()))
            cdrs = sorted(set(cdrs))
            lens = [len(x) - x.count('.') for x in cdrs]
            print(ab, icdr, sorted(Counter(lens).most_common()))
            print('\n'.join(sorted(set(cdrs))))

    if 0:
        df = pd.read_table('data/example_repertoire.tsv')
        counts = Counter(len(x) for x in df.junction_aa)
        print(sorted((x,y/df.shape[0]) for x,y in counts.most_common()))

if 0: # look at vgene distances
    import seaborn as sns
    from tcrdist.tcr_distances import compute_all_v_region_distances

    dists = compute_all_v_region_distances('human')
    vb_genes = sorted(x for x in dists if x[2]=='B' and 'OR' not in x)
    va_genes = sorted(x for x in dists if x[2]=='A' and 'OR' not in x)
    def get_gene_and_family(a):
        g = a.split('*')[0]
        num = int(g[4:].split('-')[0].split('/')[0])
        return g, g[:4]+str(num)

    seen = set()
    genes = va_genes+vb_genes
    max_d = 24.5
    #max_d = 1e6
    min_d = 0.5 #45.5
    for a1 in genes:
        g1,f1 = get_gene_and_family(a1)
        for a2 in sorted(dists[a1].keys()):
            g2,f2 = get_gene_and_family(a2)
            if g1<g2:# and f1 == f2:
                d = dists[a1][a2]
                if d<=max_d and d>=min_d and (g1,g2) not in seen:
                    seen.add((g1,g2))
                    print(d, g1, g2, a2, a2)


    # compare va to vb dists
    dfl = []
    for ab,genes in zip('AB',[va_genes, vb_genes]):
        for a1 in genes:
            g1,f1 = get_gene_and_family(a1)
            for a2 in sorted(dists[a1].keys()):
                g2,f2 = get_gene_and_family(a2)
                if g1<g2:
                    dfl.append(dict(
                        ab=ab,
                        a1=a1,
                        g1=g1,
                        a2=a2,
                        g2=g2,
                        #f1=f1,
                        dist=dists[a1][a2],
                    ))
    df = pd.DataFrame(dfl)
    df = df.sort_values('dist').drop_duplicates(['g1','g2'])
    ma = df[df.ab=='A'].dist.median()
    mb = df[df.ab=='B'].dist.median()
    plt.figure()
    sns.violinplot(data=df, x='ab', y='dist', order='AB')
    xmn,xmx = plt.xlim()
    plt.plot([xmn,xmx],[ma,ma],':k',zorder=2)
    plt.plot([xmn,xmx],[mb,mb],':k',zorder=2)
    plt.xlim((xmn,xmx))
    plt.ylim((0,plt.ylim()[1]))
    plt.title('TCRdist V gene distance distributions, alpha vs beta\n'
              'Smallest dist pair per gene-pair (ie, dropping allele info)')
    plt.tight_layout()
    pngfile = '/home/pbradley/csdat/raptcr/tmp.png'
    plt.savefig(pngfile)
    print('made:', pngfile)


if 0: # compare cdr3 distances for tcrdist vs gapped-encoding
    from raptcr.analysis import Repertoire
    import tcrdist
    from tcrdist.tcr_distances import weighted_cdr3_distance
    from raptcr.hashing import Cdr3Hasher
    from raptcr.constants.hashing import DEFAULT_DM
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import linregress
    import random


    #ks = [5,10,25,50,100,150]
    #ms = [4,8,16,32,64,128]
    ks = [10,20,40]
    radii = [6.5,12.5,24.5]
    ms = [0,3,4,5,6,7,8,12,16,32,64]

    num_pos = 16 # the length of the trimmed/gapped sequences
    SQRT = True
    plot_ms = [4,8,12,64]

    # read some epitope-specific paired tcrs
    #fname = './data/phil/conga_lit_db.tsv'
    fname = './data/phil/pregibon_tcrs.tsv'
    df = pd.read_table(fname)
    df.rename(columns={'vb':'v_call', 'ja':'j_call', 'cdr3b':'junction_aa'},
              inplace=True)

    rep = Repertoire(df)
    rep = list(rep)
    random.seed(11)
    random.shuffle(rep) # I was worried that sequence order makes a difference...

    gapped_seqs = [trim_and_gap_cdr3(cdr3, num_pos) for cdr3 in rep]
    assert all(len(x) == num_pos for x in gapped_seqs)

    if len(gapped_seqs)<40:
        # for visualization
        print('\n'.join(gapped_seqs))
        assert False

    N = len(rep)
    print('computing TCRdist cdr3 distances:', N)
    start = timer()
    # note that we are dividing by 3.0 here to remove the cdr3 weight applied in the fxn
    tcrD= np.array([weighted_cdr3_distance(x,y)/3.
                    for x in rep for y in rep]).reshape((N,N))
    print(f'computing TCRdist cdr3 distances took {timer()-start} seconds.')

    if 0: # sanity checking-- these give high correlations
        start = timer()
        tcrD2 = compute_gapped_seqs_dists(gapped_seqs)
        reg = linregress(tcrD.ravel(), tcrD2.ravel())
        print(f'compute_gapped_seqs_dists took {timer()-start} seconds.\n'
              f'correlation Rvalue: {reg.rvalue:.6f}')

        force_metric_dim = 64
        tcrD3 = compute_gapped_seqs_dists(
            gapped_seqs, force_metric_dim=force_metric_dim)
        reg = linregress(tcrD.ravel(), tcrD3.ravel())
        print(f'force_metric_dim= {force_metric_dim} correlation '
              f'Rvalue: {reg.rvalue:.6f}')

    if plot_ms:
        ncols = len(plot_ms)
        plt.figure(figsize=(6*ncols,6))
    results = []

    dist_times = []
    for m in ms:
        if m==0:
            rapD = compute_gapped_seqs_dists(gapped_seqs)
        else:
            vecs = gapped_encode_cdr3s(rep, m, num_pos, SQRT=SQRT)
            assert vecs.shape == (len(rep), m*num_pos)

            start = timer()
            rapD = squareform(pdist(vecs))
            dist_times.append(timer()-start)
            if SQRT:
                # pdist takes a sqrt at the end, but tcrdist just sums the per-position
                # scores, so we need to square the pdist distances to get agreement
                # we also need to sqrt the tcrdist distance matrix, that's done
                # inside the gapped_encode_cdr3s function
                rapD = rapD**2

        if plot_ms and m in plot_ms:
            col = plot_ms.index(m)
            plt.subplot(1, ncols, col+1)
            plt.plot(rapD.ravel(), tcrD.ravel(), 'ro', markersize=5, alpha=0.1)
            plt.xlabel('naive-gapped-MDS-embedding dist')
            plt.ylabel('TCRdist')
            mn = max(plt.xlim()[0], plt.ylim()[0])
            mx = min(plt.xlim()[1], plt.ylim()[1])
            plt.plot([mn,mx], [mn,mx], ':k')
            assert SQRT
            plt.title(f'distance comparison for m={m} naive gapped embedding')

        for k in ks:
            if k >= len(rep):
                continue
            tcrD_nbrs = get_nonself_nbrs_from_distances(tcrD, k)
            rapD_nbrs = get_nonself_nbrs_from_distances(rapD, k)

            #jaccard overlap
            nbr_overlap = len(rapD_nbrs & tcrD_nbrs)/len(rapD_nbrs | tcrD_nbrs)
            combo_nbrs_list =  list(tcrD_nbrs)+list(rapD_nbrs)
            rapD_dists = [rapD[i,j] for i,j in combo_nbrs_list]
            tcrD_dists = [tcrD[i,j] for i,j in combo_nbrs_list]

            reg = linregress(rapD_dists, tcrD_dists)
            reg2 = linregress(rapD.ravel(), tcrD.ravel())
            results.append(dict(
                m = m,
                k = k,
                SQRT=SQRT,
                nbr_overlap = nbr_overlap,
                nbrdist_rvalue = reg.rvalue,
                overall_rvalue = reg2.rvalue,
            ))
            print(results[-1])

        for radius in radii:
            tcrD_nbrs = get_nonself_nbrs_from_distances_by_radius(tcrD, radius)
            rapD_nbrs = get_nonself_nbrs_from_distances_by_radius(rapD, radius)

            #jaccard overlap
            nbr_overlap = len(rapD_nbrs & tcrD_nbrs)/len(rapD_nbrs | tcrD_nbrs)
            combo_nbrs_list =  list(tcrD_nbrs)+list(rapD_nbrs)
            rapD_dists = [rapD[i,j] for i,j in combo_nbrs_list]
            tcrD_dists = [tcrD[i,j] for i,j in combo_nbrs_list]

            reg = linregress(rapD_dists, tcrD_dists)
            reg2 = linregress(rapD.ravel(), tcrD.ravel())

            results.append(dict(
                m = m,
                radius = radius,
                SQRT=SQRT,
                nbr_overlap = nbr_overlap,
                nbrdist_rvalue = reg.rvalue,
                overall_rvalue = reg2.rvalue,
            ))
            print(results[-1])

    print(f'average dist_time= {np.mean(dist_times):.6f} w/ N= {len(rep)}')

    results = pd.DataFrame(results)
    outfile = f'cdr3dist_gapped_results_N{len(rep)}.tsv'
    results.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

    if plot_ms:
        pngfile = f'cdr3dist_gapped_comparison_N{len(rep)}.png'
        pngfile = '/home/pbradley/csdat/raptcr/'+pngfile
        plt.tight_layout()
        plt.savefig(pngfile)
        print('made:', pngfile)




if 0: # look at mds for tcrdist matrix
    # add gap at position 20
    dm = np.zeros((21,21))
    dm[:20,:20] = TCRDIST_DM
    dm[:20,20] = 4.
    dm[20,:20] = 4.

    dm = np.sqrt(dm) # take sqrt

    dims = [3,4,5,6,7,8,12,64]
    nrows, ncols = 3, len(dims)
    plt.figure(figsize=(3*ncols, 3*nrows))

    for ii,dim in enumerate(dims):
        vecs, stress = calc_mds_vecs(dm, dim, return_stress=True)

        plt.subplot(nrows, ncols, ii+1)
        plt.title(f'{dim} {stress:.4f}')
        plt.imshow(dm)
        plt.colorbar()

        plt.subplot(nrows, ncols, ncols+ii+1)
        D = squareform(pdist(vecs))
        plt.imshow(D)
        plt.colorbar()

        plt.subplot(nrows, ncols, 2*ncols+ii+1)
        plt.scatter(dm.ravel(), D.ravel(), alpha=0.2)
        plt.plot([0,2],[0,2])

    plt.tight_layout()
    pngfile = f'mds_of_tcrdist_matrix.png'
    pngfile = '/home/pbradley/csdat/raptcr/'+pngfile
    plt.savefig(pngfile)
    print('made:', pngfile)
    #plt.show()
    plt.close('all')



if 0: # compare cdr3 distances for tcrdist vs hashing
    # this was exploring the idea of creating the aa vectors for hashing
    # by using a low=dimensional MDS and then tiling those short vectors out
    # to create a length 64 (say) big vector
    # it didn't really seem to help much
    from raptcr.analysis import Repertoire
    import tcrdist
    from tcrdist.tcr_distances import weighted_cdr3_distance
    from raptcr.hashing import Cdr3Hasher
    from raptcr.constants.hashing import DEFAULT_DM
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import linregress


    #ks = [5,10,25,50,100,150]
    #ms = [4,8,16,32,64,128]
    ks = [10,20,40]
    radii = [6.5, 12.5, 24.5]
    ms = [4,8,12,16,32,64,128,256]
    #tiling_ms = list(range(2,33))+[48,64]
    #tiling_ms[0] = 0
    tiling_ms = [0]
    plot_ms = [32,64,128,256]
    SQRT = True

    # read some epitope-specific paired tcrs
    #fname = './data/phil/conga_lit_db.tsv'
    fname = './data/phil/pregibon_tcrs.tsv'
    df = pd.read_table(fname)
    df.rename(columns={'vb':'v_call', 'ja':'j_call', 'cdr3b':'junction_aa'},
              inplace=True)

    rep = Repertoire(df)

    N = df.shape[0]
    print('computing TCRdist cdr3 distances:', N)
    start = timer()
    tcrD= np.array([weighted_cdr3_distance(x,y)/3.0
                    for x in df.junction_aa for y in df.junction_aa]).reshape((N,N))
    print(f'That took {timer()-start} seconds.')


    if plot_ms:
        nrows, ncols = 2, len(plot_ms)
        plt.figure(figsize=(ncols*4, nrows*4))
    results = []
    for row, dmtag in enumerate(['DEFAULT_DM','TCRDIST_DM']):
        dm = locals()[dmtag]

        if dmtag == 'TCRDIST_DM' and SQRT:
            dm = np.sqrt(dm)

        for m in ms:
            for tiling_m in tiling_ms:
                cdr3_hasher = Cdr3Hasher(
                    m=m, distance_matrix=dm, trim_left=3, trim_right=2)
                cdr3_hasher.fit()

                if tiling_m:
                    update_hasher_aa_vectors_by_mds_tiling(dm, tiling_m, cdr3_hasher)

                vecs = cdr3_hasher.transform(rep)

                rapD = squareform(pdist(vecs))
                if SQRT:
                    rapD = rapD**2

                if m in plot_ms:
                    col = plot_ms.index(m)
                    plt.subplot(nrows, ncols, row*ncols+col+1)
                    plt.plot(rapD.ravel(), tcrD.ravel(), 'ro', markersize=5, alpha=0.1)
                    reg = linregress(rapD.ravel(), tcrD.ravel())
                    plt.xlabel('rapTCR CDR3 distance')
                    plt.ylabel('TCRdist CDR3 distance')
                    plt.title(f'rapTCR distance matrix: {dmtag}\n'
                              f'rapTCR m= {m} Rvalue= {reg.rvalue:.6f}')

                for k in ks:
                    tcrD_nbrs = get_nonself_nbrs_from_distances(tcrD, k)
                    rapD_nbrs = get_nonself_nbrs_from_distances(rapD, k)

                    #jaccard overlap
                    nbr_overlap = len(rapD_nbrs & tcrD_nbrs)/len(rapD_nbrs | tcrD_nbrs)
                    combo_nbrs_list =  list(tcrD_nbrs)+list(rapD_nbrs)
                    rapD_dists = [rapD[i,j] for i,j in combo_nbrs_list]
                    tcrD_dists = [tcrD[i,j] for i,j in combo_nbrs_list]

                    reg = linregress(rapD_dists, tcrD_dists)
                    results.append(dict(
                        dmtag = dmtag,
                        m = m,
                        k = k,
                        tiling_m = tiling_m,
                        nbr_overlap = nbr_overlap,
                        nbrdist_rvalue = reg.rvalue,
                    ))
                    print(results[-1])

    results = pd.DataFrame(results)
    outfile = f'cdr3dist_results_N{len(rep)}.tsv'
    if SQRT:
        outfile = outfile[:-4]+'_SQRT.tsv'
    results.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

    if plot_ms:
        assert tiling_ms==[0]
        pngfile = f'cdr3dist_comparison_N{len(rep)}.png'
        if SQRT:
            pngfile = pngfile.replace('.png','_SQRT.png')
        pngfile = '/home/pbradley/csdat/raptcr/'+pngfile
        plt.tight_layout()
        plt.savefig(pngfile)
        print('made:', pngfile)


if 0: # plot the naive-gapped embedding nbr overlaps
    fname = 'cdr3dist_gapped_results_N924.tsv'
    df = pd.read_table(fname)
    df = df[~df.radius.isna()]
    df = df[df.m>0] # m=0 means "exact" distances computed on the gapped seqs

    radii = sorted(set(df.radius))
    scorenames = 'nbr_overlap nbrdist_rvalue'.split()
    ms = sorted(set(df.m))


    nrows, ncols = 1, len(scorenames)

    plt.figure(figsize=(6*ncols,6*nrows))
    for col, scorename in enumerate(scorenames):
        plt.subplot(nrows, ncols, col+1)
        for radius in radii:
            mask = (df.radius==radius)
            labels = [f'{x}\n{16*x}' for x in ms]
            #labels = [f'm={x},L={16*x}' for x in ms]
            plt.plot(labels, df[mask][scorename], label=f'radius {radius}')
            #plt.xlabel('m')
            plt.xlabel('aa-embedding-dim, total-vector-len')
            plt.ylabel(scorename)
        if scorename == 'nbr_overlap':
            msg = 'Jaccard overlap in {(i,j)} non-self neighbors w/ distance<radius'
        else:
            msg = 'Pearson R-value for distances between i,j nbr pairs'
        plt.title(msg)


        plt.legend()

    plt.suptitle('Comparing TCRdist CDR3 distances to naive gapped embedding with '
                 'the length of the trimmed+gapped sequence fixed at 16\n'
                 'and various dimensions for the per-aa MDS '
                 'embedding')
    plt.tight_layout()
    pngfile = fname[:-4]+'_plots.png'
    pngfile = '/home/pbradley/csdat/raptcr/'+pngfile
    plt.savefig(pngfile)
    print('made:', pngfile)


if 0: # plot the 'tiling_m' performance
    #df = pd.read_table('cdr3dist_results_N4124.tsv')
    #df = pd.read_table('cdr3dist_results_N924.tsv')
    df = pd.read_table('cdr3dist_results_N924_SQRT.tsv')

    ks = sorted(set(df.k))
    tiling_ms = sorted(set(df.tiling_m))
    dmtags = sorted(set(df.dmtag))
    scorenames = 'nbr_overlap nbrdist_rvalue'.split()

    nrows, ncols = len(scorenames), len(ks)

    plt.figure(figsize=(4*ncols,4*nrows))
    for row, scorename in enumerate(scorenames):
        for col, k in enumerate(ks):
            plt.subplot(nrows, ncols, row*ncols+col+1)

            for dmtag in dmtags:
                mask = (df.dmtag==dmtag)&(df.k==k)
                plt.plot(df[mask].tiling_m, df[mask][scorename], label=dmtag)
            plt.xlabel('tiling_m')
            plt.ylabel(scorename)
            plt.legend()
            plt.title(f'k= {k}')
    plt.tight_layout()
    plt.show()
    plt.close('all')

if 0: # plot setup performance of PynndescentIndex
    vals = np.array([[  10000,   1.113],
                     [  50000,   6.510],
                     [ 100000,  13.075],
                     [ 500000,  74.740],
                     [1500000, 249.026],
                     ])
    plt.figure()
    plt.plot(vals[:,0], vals[:,1])
    plt.scatter(vals[:,0], vals[:,1])
    xn = vals[-1,0]
    for v0,v1 in zip(vals[:-1], vals[1:]):
        x0,y0=v0
        x1,y1=v1
        slope = (y1-y0)/(x1-x0)
        yn = y1 + slope*(xn-x1)
        plt.plot([x1,xn], [y1,yn], ':k')

    plt.xlabel('num TCRs')
    plt.ylabel('seconds')
    plt.title('PynndescentIndex add(...) time')
    plt.show()




if 0: # try some indexing
    from raptcr.analysis import Repertoire
    from raptcr.hashing import Cdr3Hasher
    from raptcr.indexing import FlatIndex, PynndescentIndex

    start = timer()
    if 0:
        fname = '/home/pbradley/csdat/tcrpepmhc/amir/pregibon_test2/tcr_db.tsv'
        df = pd.read_table(fname)
        df.rename(columns={'vb':'v_call', 'ja':'j_call', 'cdr3b':'junction_aa'},
                  inplace=True)
    if 2:
        fname = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2022-01-22.tsv'
        print('reading:', fname)
        df = pd.read_table(fname)
        df.rename(columns={'vb':'v_call', 'ja':'j_call', 'cdr3b':'junction_aa'},
                  inplace=True)
    print('reading took:', timer() - start)

    rep = Repertoire(df.head(1500000))

    cdr3_hasher = Cdr3Hasher(m=64)
    cdr3_hasher.fit()

    #print('transform:', rep)
    #vecs = cdr3_hasher.transform(rep)


    # initialize index and add data
    print('create index')
    if 0:
        index = FlatIndex(hasher=cdr3_hasher)
        ks = [5,10,15,20,25,100,250]
    else:
        ks = [25]
        index = PynndescentIndex(hasher=cdr3_hasher, k=ks[0])

    print('add the data')
    start = timer()
    index.add(rep)
    print('adding took:', timer() - start)

    query_seqs = list(rep)[:1000]#['CASSKRDRGNGGYTF', "CASSITPGQGTDEQYF"]
    for k in ks:
        print('find',k,'neighbors for m=', len(query_seqs), 'vs N=', len(rep))
        start = timer()
        if len(ks)==1:
            knn_result = index.knn_search()#query_seqs)
        else:
            knn_result = index.knn_search(query_seqs, k=k)
        print('searching took:', timer() - start, k)

    print('DONE')


if 0: # look at blosum distance matrix
    import raptcr.constants.hashing
    import raptcr.constants.base
    import seaborn as sns
    from sklearn.manifold import MDS

    aas = np.array(list(raptcr.constants.base.AALPHABET))
    #dm = raptcr.constants.hashing.DEFAULT_DM
    dm = np.sqrt(TCRDIST_DM) # since we add the values from one pos to the next...
    # seems like a sqrt is appropriate here
    #

    if 0: # show
        plt.figure()
        plt.imshow(dm)
        plt.colorbar()
        plt.xticks(np.arange(20), aas)
        plt.yticks(np.arange(20), aas)
        plt.show()

    if 0: # clustermap the distances
        cm = sns.clustermap(dm)
        inds = cm.dendrogram_row.reordered_ind
        print(aas[inds])
        cm.ax_heatmap.set_yticks(np.arange(20)+0.5)
        cm.ax_heatmap.set_yticklabels(aas[inds])#, rotation=0, fontsize=6);
        inds = cm.dendrogram_col.reordered_ind
        print(aas[inds])
        cm.ax_heatmap.set_xticks(np.arange(20)+0.5)
        cm.ax_heatmap.set_xticklabels(aas[inds])
        plt.show()


    if 2:# check triangle ineq
        for i, a in enumerate(aas):
            print(i,a)
            for j, b in enumerate(aas):
                for k, c in enumerate(aas):
                    if dm[i,k] > dm[i,j] + dm[j,k] +1e-3:
                        print('trifail:',a,b,c, dm[i,k], dm[i,j] + dm[j,k])


    if 0: # try mds
        m = 64
        print('running mds 64')
        mds = MDS(n_components=m, dissimilarity="precomputed", random_state=11,
                  normalized_stress=False)
        vecs = mds.fit_transform(dm)
        print('mds:', mds.stress_)

        dims = np.arange(1,64)
        stressl = []
        for ii in dims:
            #print('running mds', ii)
            mds3 = MDS(n_components=ii, dissimilarity="precomputed", random_state=11,
                       normalized_stress=False)
            vecs3 = mds3.fit_transform(dm)
            print('mds:', ii, mds3.stress_)
            stressl.append(mds3.stress_)


        plt.figure()
        plt.plot(np.sqrt(dims), np.sqrt(stressl))
        plt.scatter(np.sqrt(dims), np.sqrt(stressl))
        locs = plt.xticks()[0]
        plt.xticks(locs, [x**2 for x in locs])
        locs = plt.yticks()[0]
        plt.yticks(locs, [x**2 for x in locs])
        plt.xlabel('embedding dim')
        plt.ylabel('stress')
        plt.show()




