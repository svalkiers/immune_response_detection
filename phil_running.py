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
import time

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True)
parser.add_argument('--radius', type=float)
parser.add_argument('--sleeptime', type=float, default=0)
parser.add_argument('--fg_filename')
parser.add_argument('--bg_filename')
parser.add_argument('--bg_filenames', nargs='*')
parser.add_argument('--outfile_prefix')
parser.add_argument('--outfile')
parser.add_argument('--max_tcrs', type=int)
parser.add_argument('--num_bg_tcrs', type=int)
parser.add_argument('--num_nbrs', type=int)
parser.add_argument('--maxdist', type=int)
parser.add_argument('--bg_nums', nargs='*', type=int) # which bg reps to use
parser.add_argument('--aa_mds_dim', type=int, default=8)
parser.add_argument('--filename')
parser.add_argument('--skip_fg', action='store_true')
parser.add_argument('--start_index', type=int)
parser.add_argument('--stop_index', type=int)

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
        bg_nums = list(range(8))

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
    # parsed_df = parse_cdr3_aa_regions(
    #     tcrs, organism, chain, v_column, j_column, cdr3_column)
    # parsed_df_x1 = parse_cdr3_aa_regions(
    #     tcrs, organism, chain, v_column, j_column, cdr3_column, extend_align=1)

    junctions = parse_junctions_for_background_resampling(
        tcrs, organism, chain, v_column, j_column, cdr3_column, 'cdr3nt')
    # tcr_tuples = [(None,x) for x in tcrs['v j cdr3aa cdr3nt'.split()].itertuples(
    #     name=None, index=None)]
    # junctions = tcrdist.tcr_sampler.parse_tcr_junctions(organism, tcr_tuples)
    # junctions = add_vdj_splits_info_to_junctions(junctions)

    if not args.skip_fg: # fg radius search:
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
        # if bgnum==0:
        #     bg_tcrs = resample_cdr3_aa_regions(parsed_df)
        # elif bgnum==1:
        #     bg_tcrs = resample_cdr3_aa_regions(parsed_df_x1)
        # elif bgnum==2:
        #     bg_tcrs = resample_cdr3_aa_regions(
        #         parsed_df, match_j_families=True)
        # elif bgnum==3:
        #     bg_tcrs = resample_cdr3_aa_regions(
        #         parsed_df_x1, match_j_families=True)
        if bgnum==4:
            bg_tcr_tuples = tcrdist.tcr_sampler.resample_shuffled_tcr_chains(
                organism, tcrs.shape[0], chain, junctions)
            bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])
        # elif bgnum==5:
        #     bg_tcr_tuples = resample_cdr3_nt_regions(junctions) # NEW!
        #     bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])
        elif bgnum==6:
            bg_tcr_tuples = resample_background_tcrs_v4(organism, chain, junctions)
            bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])
        elif bgnum==7:
            bg_tcrs= sample_igor_tcrs(tcrs.shape[0], v_column='v', cdr3aa_column='cdr3')

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
    # parsed_df = parse_cdr3_aa_regions(
    #     tcrs, organism, chain, v_column, j_column, cdr3_column)
    # parsed_df_x1 = parse_cdr3_aa_regions(
    #     tcrs, organism, chain, v_column, j_column, cdr3_column, extend_align=1)

    # tcr_tuples = [(None,x) for x in tcrs['v j cdr3aa cdr3nt'.split()].itertuples(
    #     name=None, index=None)]
    # junctions = tcrdist.tcr_sampler.parse_tcr_junctions(organism, tcr_tuples)
    # junctions = add_vdj_splits_info_to_junctions(junctions)
    junctions = parse_junctions_for_background_resampling(
        tcrs, organism, chain, v_column, j_column, cdr3_column, 'cdr3nt')

    if not args.skip_fg: # fg radius search:
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
        # if bgnum==0:
        #     bg_tcrs = resample_cdr3_aa_regions(parsed_df)
        # elif bgnum==1:
        #     bg_tcrs = resample_cdr3_aa_regions(parsed_df_x1)
        # elif bgnum==2:
        #     bg_tcrs = resample_cdr3_aa_regions(
        #         parsed_df, match_j_families=True)
        # elif bgnum==3:
        #     bg_tcrs = resample_cdr3_aa_regions(
        #         parsed_df_x1, match_j_families=True)
        if bgnum==4:
            bg_tcr_tuples = tcrdist.tcr_sampler.resample_shuffled_tcr_chains(
                organism, tcrs.shape[0], chain, junctions)
            bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])
        elif bgnum==6:
            bg_tcr_tuples = resample_background_tcrs_v4(organism, chain, junctions)
            bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])
        # elif bgnum==5:
        #     bg_tcr_tuples = resample_cdr3_nt_regions(junctions) # NEW!
        #     bg_tcrs = pd.DataFrame([dict(v=x[0], cdr3=x[2]) for x in bg_tcr_tuples])

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


elif args.mode == 'paired_ranges':
    sleeptime = args.sleeptime * random.random()
    print('sleeping:', sleeptime)
    time.sleep(sleeptime)


    organism = 'human'
    print('reading:', args.filename)
    tcrs = pd.read_table(args.filename)
    num_tcrs = tcrs.shape[0]
    print('num_tcrs:', num_tcrs)

    # assumes that the file is already filtered
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')
    assert tcrs.shape[0] == num_tcrs

    fg_avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', args.aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    fg_bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', args.aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    # compute nearest neighbors in fg paired vecs
    fg_vecs = np.hstack([fg_avecs, fg_bvecs])
    assert fg_vecs.shape == (tcrs.shape[0], fg_avecs.shape[1] + fg_bvecs.shape[1])

    qvecs = fg_vecs[args.start_index:args.stop_index]

    print('start IndexFlatL2 range search', qvecs.shape, fg_vecs.shape)
    start = timer()
    idx = faiss.IndexFlatL2(fg_vecs.shape[1])
    idx.add(fg_vecs)
    lims,D,I = idx.range_search(qvecs, args.radius)
    print(f'IndexFlatL2 range search took {timer()-start:.2f}')
    outprefix = (f'{args.outfile_prefix}_{args.radius:.1f}_{qvecs.shape[0]}_'
                 f'{fg_vecs.shape[0]}_{fg_vecs.shape[1]}')

    np.save(f'{outprefix}_lims.npy', lims)
    np.save(f'{outprefix}_D.npy', D)
    np.save(f'{outprefix}_I.npy', I)


elif args.mode == 'paired_knns':
    sleeptime = args.sleeptime * random.random()
    print('sleeping:', sleeptime)
    time.sleep(sleeptime)

    organism = 'human'
    print('reading:', args.filename)
    tcrs = pd.read_table(args.filename)
    num_tcrs = tcrs.shape[0]
    print('num_tcrs:', num_tcrs)

    # assumes that the file is already filtered
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')
    assert tcrs.shape[0] == num_tcrs

    fg_avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', args.aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    fg_bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', args.aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    # compute nearest neighbors in fg paired vecs
    fg_vecs = np.hstack([fg_avecs, fg_bvecs])
    assert fg_vecs.shape == (tcrs.shape[0], fg_avecs.shape[1] + fg_bvecs.shape[1])

    qvecs = fg_vecs[args.start_index:args.stop_index]

    print('start IndexFlatL2 knn search', args.num_nbrs, qvecs.shape, fg_vecs.shape)
    start = timer()
    idx = faiss.IndexFlatL2(fg_vecs.shape[1])
    idx.add(fg_vecs)
    D, I = idx.search(qvecs, args.num_nbrs)
    print(f'IndexFlatL2 knn search took {timer()-start:.2f}')
    outprefix = (f'{args.outfile_prefix}_{args.num_nbrs}_{qvecs.shape[0]}_'
                 f'{fg_vecs.shape[0]}_{fg_vecs.shape[1]}')

    np.save(f'{outprefix}_D.npy', D)
    np.save(f'{outprefix}_I.npy', I)


elif args.mode == 'paired_backgrounds':
    sleeptime = args.sleeptime * random.random()
    print('sleeping:', sleeptime)
    time.sleep(sleeptime)

    organism = 'human'
    print('reading:', args.fg_filename)
    tcrs = pd.read_table(args.fg_filename)
    num_tcrs = tcrs.shape[0]
    print('num_tcrs:', num_tcrs)

    # read the background tcrs
    bg_tcrs = pd.read_table(args.bg_filename).head(args.num_bg_tcrs)

    # assumes that the file is already filtered
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')
    assert tcrs.shape[0] == num_tcrs
    bg_tcrs = filter_out_bad_genes_and_cdr3s(
        bg_tcrs, 'va', 'cdr3a', organism, 'A', j_column='ja')
    bg_tcrs = filter_out_bad_genes_and_cdr3s(
        bg_tcrs, 'vb', 'cdr3b', organism, 'B', j_column='jb')
    assert bg_tcrs.shape[0] == args.num_bg_tcrs
    print('num_bg_tcrs:', bg_tcrs.shape[0])

    # subset to desired range
    tcrs = tcrs.iloc[args.start_index:args.stop_index]

    fg_avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', args.aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    fg_bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', args.aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    bg_avecs = gapped_encode_tcr_chains(
        bg_tcrs, organism, 'A', args.aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    bg_bvecs = gapped_encode_tcr_chains(
        bg_tcrs, organism, 'B', args.aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    start = timer()
    ab_counts = compute_background_paired_tcrdist_distributions(
        fg_avecs, fg_bvecs, bg_avecs, bg_bvecs, args.maxdist)
    print(f'paired distribution calc took {timer()-start:.2f}')

    np.save(args.outfile, ab_counts)

elif args.mode == 'vector_knns':
    sleeptime = args.sleeptime * random.random()
    print('sleeping:', sleeptime)
    time.sleep(sleeptime)

    organism = 'human'
    print('reading:', args.filename)
    fg_vecs = np.load(args.filename)

    qvecs = fg_vecs[args.start_index:args.stop_index]

    print('start IndexFlatL2 knn search', args.num_nbrs, qvecs.shape, fg_vecs.shape)
    start = timer()
    idx = faiss.IndexFlatL2(fg_vecs.shape[1])
    idx.add(fg_vecs)
    D, I = idx.search(qvecs, args.num_nbrs)
    print(f'IndexFlatL2 knn search took {timer()-start:.2f}')
    outprefix = (f'{args.outfile_prefix}_{args.num_nbrs}_{args.start_index}_'
                 f'{args.stop_index}_{fg_vecs.shape[0]}_{fg_vecs.shape[1]}')

    np.save(f'{outprefix}_D.npy', D)
    np.save(f'{outprefix}_I.npy', I)


else:
    print('unrecognized mode:', args.mode)

print('DONE')
