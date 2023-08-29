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
from raptcr.constants.modules import tcrdist
import time
print('tcrdist:', tcrdist)

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
parser.add_argument('--v_column', default='v')
parser.add_argument('--j_column', default='j')
parser.add_argument('--cdr3aa_column', default='cdr3aa')
parser.add_argument('--cdr3nt_column', default='cdr3nt')
parser.add_argument('--chain', default='B')
parser.add_argument('--organism', default='human')
parser.add_argument('--filetag')
parser.add_argument('--fg_runtag')
parser.add_argument('--bg_runtag')
parser.add_argument('--min_fg_bg_nbr_ratio', type=float, default=2.0)
parser.add_argument('--clobber', action='store_true')

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


elif args.mode == 'make_bg': # make and save background file
    assert args.outfile

    tcrs = pd.read_table(args.filename)
    # tcrs.rename(columns={
    #     args.v_column:'v',
    #     args.j_column:'j',
    #     args.cdr3aa_column:'cdr3aa',
    #     args.cdr3nt_column:'cdr3nt',
    #     }, inplace=True)

    #v_column, j_column, cdr3_column = 'v','j','cdr3aa'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, args.v_column, args.cdr3aa_column, args.organism, args.chain,
        j_column=args.j_column)
    tcrs[args.cdr3nt_column] = tcrs[args.cdr3nt_column].str.lower()

    junctions = parse_junctions_for_background_resampling(
        tcrs, args.organism, args.chain, args.v_column, args.j_column,
        args.cdr3aa_column, args.cdr3nt_column)

    bg_tcr_tuples = resample_background_tcrs_v4(args.organism, args.chain, junctions)
    bg_tcrs = pd.DataFrame([{args.v_column:x[0], args.j_column:x[1],
                             args.cdr3aa_column:x[2], args.cdr3nt_column:x[3]}
                            for x in bg_tcr_tuples])

    bg_tcrs.to_csv(args.outfile, sep='\t', index=False)
    print('made:', args.outfile)

elif args.mode == 'nbr_counts': # count nbrs
    assert args.outfile

    start0 = timer()
    fg_tcrs = pd.read_table(args.fg_filename)

    fg_tcrs = filter_out_bad_genes_and_cdr3s(
        fg_tcrs, args.v_column, args.cdr3aa_column, args.organism, args.chain,
        j_column=args.j_column)

    fg_vecs = gapped_encode_tcr_chains(
        fg_tcrs, args.organism, args.chain, args.aa_mds_dim, v_column=args.v_column,
        cdr3_column=args.cdr3aa_column).astype(np.float32)

    print('num_fg_tcrs:', fg_vecs.shape[0], args.fg_filename)

    nbr_counts = np.zeros((fg_vecs.shape[0],))#, dtype=int)

    for bg_filename in args.bg_filenames:
        bg_tcrs = pd.read_table(bg_filename)

        bg_tcrs = filter_out_bad_genes_and_cdr3s(
            bg_tcrs, args.v_column, args.cdr3aa_column, args.organism, args.chain,
            j_column=args.j_column)

        bg_vecs = gapped_encode_tcr_chains(
            bg_tcrs, args.organism, args.chain, args.aa_mds_dim, v_column=args.v_column,
            cdr3_column=args.cdr3aa_column).astype(np.float32)

        print('num_bg_tcrs:', bg_vecs.shape[0], bg_filename)

        idx = faiss.IndexFlatL2(bg_vecs.shape[1])
        idx.add(bg_vecs)
        print('range_searching:', fg_vecs.shape[0], bg_vecs.shape[0])
        start = timer()
        lims,D,I = idx.range_search(fg_vecs, args.radius)
        print(f'range_search took {timer()-start:.2f} secs total_time: '
              f'{timer()-start0:.2f}, {fg_vecs.shape[0]} x {bg_vecs.shape[0]}')
        nbr_counts += (lims[1:] - lims[:-1])

    np.save(args.outfile, nbr_counts)
    print('made:', args.outfile)

elif args.mode == 't1d_step2':
    # accumulate the foreground and background nbr-counts
    # calculate hypergeometric pvals
    # e.g. filetag = 'cohort_1_Brusko_9964_TCRB_imgt'
    slurmdir = '/home/pbradley/csdat/raptcr/slurm/'
    fg_counts_file = f'{slurmdir}{args.fg_runtag}/{args.filetag}_nbr_totals.npy'
    bg_counts_file = f'{slurmdir}{args.bg_runtag}/{args.filetag}_nbr_totals.npy'
    pvals_outfile = f'{slurmdir}{args.fg_runtag}/{args.filetag}_pvals.tsv'

    if exists(pvals_outfile) and not args.clobber:
        print(f'{pvals_outfile} already exists and no --clobber, stopping')
        print('DONE')
        exit()

    cohort = int(args.filetag.split('_')[1])
    EXPECTED_NUM_FILES = {1:1425}[cohort] # add other cohorts here!
    BATCH_SIZE = 20

    bdir = '/fh/fast/bradley_p/t1d/'

    all_totals = {}
    for line in open(f'{bdir}wc_cohort_{cohort}_tsvs.txt','r'):
        l = line.split()
        if l[1] == 'total':
            continue
        ftag = f'cohort_{cohort}_'+l[1].split('/')[1][:-4]
        all_totals[ftag] = int(l[0])-1 # drop header line
    num_files = len(all_totals)
    assert num_files == EXPECTED_NUM_FILES
    num_batches = (num_files-1)//BATCH_SIZE + 1
    total_bg_tcrs = sum(all_totals.values())

    # accumulate the counts:
    for runtag, counts_file in [[args.fg_runtag, fg_counts_file],
                                [args.bg_runtag, bg_counts_file]]:
        if not exists(counts_file): # make the counts file
            print(f'need to create {counts_file} from {num_batches} countsfiles')
            mydir = f'{slurmdir}{runtag}/{args.filetag}/'
            nbr_counts = None
            missed = []
            for b in range(num_batches):
                start, stop = b*BATCH_SIZE, (b+1)*BATCH_SIZE
                nbrsfile = f'{mydir}{args.filetag}_{b}_{start}_{stop}.npy'
                if not exists(nbrsfile):
                    missed.append(nbrsfile)
                    print('ERROR: missing', nbrsfile)
                    continue
                #assert exists(nbrsfile)
                counts = np.load(nbrsfile)
                if b==0:
                    nbr_counts = counts
                else:
                    nbr_counts += counts
            if missed:
                print(f'ERROR missing {len(missed)} counts files so cant make',
                      counts_file)
                print('missed', ' '.join(missed))
                sys.stderr.write('ERROR missing some counts files for {counts_file}\n')
                exit() # NOTE EARLY EXIT WITHOUT "DONE"

            np.save(counts_file, nbr_counts)
            print('made:', counts_file)

    # now compute the p-values
    ftag = '_'.join(args.filetag.split('_')[2:]) # remove the 'cohort_1_' part
    repfile = f'{bdir}cohort_{cohort}/{ftag}.tsv'
    assert exists(repfile)

    tcrs = pd.read_table(repfile)
    num_tcrs = tcrs.shape[0]

    assert num_tcrs == all_totals[args.filetag]

    fg_nbr_counts = np.load(fg_counts_file)
    bg_nbr_counts = np.load(bg_counts_file)

    assert fg_nbr_counts.shape == bg_nbr_counts.shape == (num_tcrs,) # could fail??

    assert np.sum(fg_nbr_counts<0.5)==0 # no zeros
    fg_nbr_counts -= 1. # self nbrs

    pvals = compute_nbr_count_pvalues(
        fg_nbr_counts, bg_nbr_counts, total_bg_tcrs,
        num_fg_tcrs = total_bg_tcrs,
        min_fg_bg_nbr_ratio = args.min_fg_bg_nbr_ratio,
    )

    pvals = pvals.join(tcrs, on='tcr_index')

    pvals.to_csv(pvals_outfile, sep='\t', index=False)
    print('made:', pvals_outfile)

else:
    print('unrecognized mode:', args.mode)

print('DONE')
