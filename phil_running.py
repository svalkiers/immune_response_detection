######################################################################################88
# this is a script for running bigger calculations on the cluster
#
import raptcr
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

from phil_functions import * # bad, temporary...

import faiss
import argparse
import tcrdist

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True)
parser.add_argument('--radius', type=float)
parser.add_argument('--fg_filename')
parser.add_argument('--bg_filenames', nargs='*')
parser.add_argument('--outfile_prefix')
parser.add_argument('--max_tcrs', type=int)
parser.add_argument('--num_nbrs', type=int)
parser.add_argument('--bg_nums', nargs='*', type=int) # which bg reps to use
parser.add_argument('--aa_mds_dim', type=int, default=8)
parser.add_argument('--filename')

args = parser.parse_args()

if args.mode == 'brit_vs_brit': # britanov-vs-britanova neighbor calculation

    # load data from the Britanova aging study; I downloaded the files from:
    # https://zenodo.org/record/826447#.Y-7Ku-zMIWo
    print('reading:', args.fg_filename)

    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = read_britanova_tcrs(args.fg_filename, max_tcrs=args.max_tcrs)

    # encode the tcrs
    vecs = gapped_encode_tcr_chains(
        tcrs, organism, chain, args.aa_mds_dim, v_column=v_column,
        cdr3_column=cdr3_column).astype(np.float32)

    for bg_filename in args.bg_filenames:
        bg_tag = bg_filename.split('/')[-1][:-3] # for the outfile

        bg_tcrs = read_britanova_tcrs(bg_filename, max_tcrs=args.max_tcrs)

        bg_vecs = gapped_encode_tcr_chains(
            bg_tcrs, organism, chain, args.aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column).astype(np.float32)

        idx = faiss.IndexFlatL2(bg_vecs.shape[1])
        idx.add(bg_vecs)
        start = timer()
        lims,D,I = idx.range_search(vecs, args.radius)
        print(f'bg range_search took {timer()-start:.2f} secs', len(vecs))
        bg_nbr_counts = lims[1:]-lims[:-1]

        outfile = f'{args.outfile_prefix}_bg_{bg_tag}_nbr_counts.npy'
        np.save(outfile, bg_nbr_counts)
        print('made:', outfile)

elif args.mode == 'brit_vs_bg': # britanova vs background searches

    bg_nums = args.bg_nums
    if bg_nums is None:
        bg_nums = list(range(6))

    # load data from the Britanova aging study; I downloaded the files from:
    # https://zenodo.org/record/826447#.Y-7Ku-zMIWo
    #
    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = read_britanova_tcrs(args.filename, max_tcrs=args.max_tcrs)

    # encode the tcrs
    vecs = gapped_encode_tcr_chains(
        tcrs, organism, chain, args.aa_mds_dim, v_column=v_column,
        cdr3_column=cdr3_column).astype(np.float32)

    # parse repertoire, create background reps
    parsed_df = parse_cdr3_aa_regions(
        tcrs, organism, chain, v_column, j_column, cdr3_column)
    parsed_df_x1 = parse_cdr3_aa_regions(
        tcrs, organism, chain, v_column, j_column, cdr3_column, extend_align=1)

    tcr_tuples = [(None,x) for x in tcrs['v j cdr3aa cdr3nt'.split()].itertuples(
        name=None, index=None)]
    junctions = tcrdist.tcr_sampler.parse_tcr_junctions(organism, tcr_tuples)
    junctions = add_vdj_splits_info_to_junctions(junctions)

    # fg radius search:
    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)
    start = timer()
    lims,D,I = idx.range_search(vecs, args.radius)
    print(f'fg range_search took {timer()-start:.2f} secs', len(vecs))
    nbr_counts = lims[1:]-lims[:-1] - 1 # exclude self

    outfile = f'{args.outfile_prefix}_fg_nbr_counts.npy'
    np.save(outfile, nbr_counts)
    print('made:', outfile, flush=True)


    for bgnum in bg_nums:
        if bgnum==0:
            bg_tcrs = resample_cdr3_aa_regions(parsed_df)
        elif bgnum==1:
            bg_tcrs = resample_cdr3_aa_regions(parsed_df_x1)
        elif bgnum==2:
            bg_tcrs = resample_cdr3_aa_regions(
                parsed_df, match_j_families=True)
        elif bgnum==3:
            bg_tcrs = resample_cdr3_aa_regions(
                parsed_df_x1, match_j_families=True)
        elif bgnum==4:
            bg_tcr_tuples = tcrdist.tcr_sampler.resample_shuffled_tcr_chains(
                organism, tcrs.shape[0], chain, junctions)
            bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])
        elif bgnum==5:
            bg_tcr_tuples = resample_cdr3_nt_regions(junctions) # NEW!
            bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])

        bg_tcrs.rename(columns={'v':v_column, 'cdr3':cdr3_column},
                       inplace=True)

        bg_vecs = gapped_encode_tcr_chains(
            bg_tcrs, organism, chain, args.aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column).astype(np.float32)

        idx = faiss.IndexFlatL2(bg_vecs.shape[1])
        idx.add(bg_vecs)
        start = timer()
        lims,D,I = idx.range_search(vecs, args.radius)
        print(f'bg range_search took {timer()-start:.2f} secs', len(vecs))
        bg_nbr_counts = lims[1:]-lims[:-1]

        outfile = f'{args.outfile_prefix}_bg_{bgnum}_nbr_counts.npy'
        np.save(outfile, bg_nbr_counts)
        print('made:', outfile)

elif args.mode == 'nndists_vs_bg': # britanova vs background searches
    # compute nndist as mean of distances to --num_nbrs nearest (nonself) nbrs

    bg_nums = args.bg_nums
    if bg_nums is None:
        bg_nums = list(range(6))

    # load data from the Britanova aging study; I downloaded the files from:
    # https://zenodo.org/record/826447#.Y-7Ku-zMIWo
    #
    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = read_britanova_tcrs(args.filename, max_tcrs=args.max_tcrs)

    # encode the tcrs
    vecs = gapped_encode_tcr_chains(
        tcrs, organism, chain, args.aa_mds_dim, v_column=v_column,
        cdr3_column=cdr3_column).astype(np.float32)

    # parse repertoire, create background reps
    parsed_df = parse_cdr3_aa_regions(
        tcrs, organism, chain, v_column, j_column, cdr3_column)
    parsed_df_x1 = parse_cdr3_aa_regions(
        tcrs, organism, chain, v_column, j_column, cdr3_column, extend_align=1)

    tcr_tuples = [(None,x) for x in tcrs['v j cdr3aa cdr3nt'.split()].itertuples(
        name=None, index=None)]
    junctions = tcrdist.tcr_sampler.parse_tcr_junctions(organism, tcr_tuples)
    junctions = add_vdj_splits_info_to_junctions(junctions)

    # fg radius search:
    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)
    start = timer()

    D,I = idx.search(vecs, args.num_nbrs+1)
    nndists = np.mean(D[:,1:], axis=-1)
    print(f'knn search took {timer()-start:.2f} secs', len(vecs))

    outfile = f'{args.outfile_prefix}_fg_nndists.npy'
    np.save(outfile, nndists)
    print('made:', outfile, flush=True)


    for bgnum in bg_nums:
        if bgnum==0:
            bg_tcrs = resample_cdr3_aa_regions(parsed_df)
        elif bgnum==1:
            bg_tcrs = resample_cdr3_aa_regions(parsed_df_x1)
        elif bgnum==2:
            bg_tcrs = resample_cdr3_aa_regions(
                parsed_df, match_j_families=True)
        elif bgnum==3:
            bg_tcrs = resample_cdr3_aa_regions(
                parsed_df_x1, match_j_families=True)
        elif bgnum==4:
            bg_tcr_tuples = tcrdist.tcr_sampler.resample_shuffled_tcr_chains(
                organism, tcrs.shape[0], chain, junctions)
            bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])
        elif bgnum==5:
            bg_tcr_tuples = resample_cdr3_nt_regions(junctions) # NEW!
            bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])

        bg_tcrs.rename(columns={'v':v_column, 'cdr3':cdr3_column},
                       inplace=True)

        bg_vecs = gapped_encode_tcr_chains(
            bg_tcrs, organism, chain, args.aa_mds_dim, v_column=v_column,
            cdr3_column=cdr3_column).astype(np.float32)

        idx = faiss.IndexFlatL2(bg_vecs.shape[1])
        idx.add(bg_vecs)
        start = timer()
        D,I = idx.search(vecs, args.num_nbrs)
        print(f'bg knn search took {timer()-start:.2f} secs', len(vecs))
        bg_nndists = np.mean(D, axis=-1)

        outfile = f'{args.outfile_prefix}_bg_{bgnum}_nndists.npy'
        np.save(outfile, bg_nndists)
        print('made:', outfile)
else:
    print('unrecognized mode:', args.mode)

