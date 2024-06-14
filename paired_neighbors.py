######################################################################################88
import argparse
parser = argparse.ArgumentParser(
    description="find paired tcr neighbor enrichment",
    epilog='''

Example command line:

python paired_neighbors.py --tcrs all_uniq_tcrs.tsv --organism human --outfile all_nbr_pvals.tsv

''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--tcrs', required=True, help='tcr input file')
parser.add_argument('--aa_mds_dim', type=int, default=8)
parser.add_argument('--bg_multiplier', type=int, default=10)
parser.add_argument('--outfile', required=True)
parser.add_argument('--organism', required=True, choices=['human','mouse'])

args = parser.parse_args()

import faiss
import itertools as it
import numpy as np
import pandas as pd
from timeit import default_timer as timer
#from collections import Counter
#from sys import exit
#import sys
#from os.path import exists


from raptcr.constants.modules.tcrdist.tcr_sampler import (
    parse_tcr_junctions,
    find_alternate_alleles_for_tcrs,
)

from phil_functions import (
    resample_background_tcrs_v4,
    filter_out_bad_genes_and_cdr3s,
    gapped_encode_tcr_chains,
    compute_background_paired_tcrdist_distributions,
)


def make_background_tcrs(tcrs, organism, multiplier=10, fix_alleles=True):
    ''' make Nx background TCRs where N= multiplier is an integer>=1

    might not be exactly Nx if some TCRs get filtered...

    tcrs should already have been filtered
    '''


    acols = 'va ja cdr3a cdr3a_nucseq'.split()
    bcols = 'vb jb cdr3b cdr3b_nucseq'.split()
    atcr_tuples = tcrs[acols].itertuples(name=None, index=None)
    btcr_tuples = tcrs[bcols].itertuples(name=None, index=None)

    if fix_alleles:
        tcr_tuples_fixed = find_alternate_alleles_for_tcrs(
            organism, list(zip(atcr_tuples, btcr_tuples)), verbose=True)
        atcr_tuples =  [x[0] for x in tcr_tuples_fixed]
        btcr_tuples =  [x[1] for x in tcr_tuples_fixed]

    junctions = parse_tcr_junctions('human', list(zip(atcr_tuples, btcr_tuples)))

    dfl = []
    for r in range(multiplier):
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
        dfl.append(bg_tcrs)

    bg_tcrs = pd.concat(dfl).reset_index(drop=True)
    return bg_tcrs


def get_background_nbr_counts(
        tcrs,
        bg_tcrs,
        organism,
        aa_mds_dim=8,
        maxdist=96,
):
    ''' Compute the background paired tcrdist distribution by taking the
    convolution of the alpha and beta single-chain tcrdist distributions.
    The effective number of paired background comparisons is len(bg_tcrs)**2

    returns an integer-valued numpy array of shape (num_fg_tcrs, maxdist+1)

    histogram bin-size is 1.0

    first bin is (-0.5, 0.5), last bin is (maxdist-0.5, maxdist+0.5)

    tcrs and bg_tcrs should already have been filtered

    (see phil_functions.compute_background_paired_tcrdist_distributions)
    '''

    fg_avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    fg_bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    bg_avecs = gapped_encode_tcr_chains(
        bg_tcrs, organism, 'A', aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    bg_bvecs = gapped_encode_tcr_chains(
        bg_tcrs, organism, 'B', aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    start = timer()
    ab_counts = compute_background_paired_tcrdist_distributions(
        fg_avecs, fg_bvecs, bg_avecs, bg_bvecs, maxdist)
    print(f'paired distribution calc took {timer()-start:.2f}')

    return ab_counts



def get_foreground_nbr_counts(
        tcrs,
        organism,
        aa_mds_dim=8,
        radius=96.5,
):
    '''
    Get the faiss data (lims,D,I) for range_search of tcrs against self

    so it will include self-distances

    tcrs and bg_tcrs should already have been filtered
    '''

    fg_avecs = gapped_encode_tcr_chains(
        tcrs, organism, 'A', aa_mds_dim, v_column='va',
        cdr3_column='cdr3a').astype(np.float32)

    fg_bvecs = gapped_encode_tcr_chains(
        tcrs, organism, 'B', aa_mds_dim, v_column='vb',
        cdr3_column='cdr3b').astype(np.float32)

    # compute nearest neighbors in fg paired vecs
    fg_vecs = np.hstack([fg_avecs, fg_bvecs])
    assert fg_vecs.shape == (tcrs.shape[0], fg_avecs.shape[1] + fg_bvecs.shape[1])

    qvecs = fg_vecs # could optionally split into batches

    print('start IndexFlatL2 range search', qvecs.shape, fg_vecs.shape)
    start = timer()
    idx = faiss.IndexFlatL2(fg_vecs.shape[1])
    idx.add(fg_vecs)
    lims, D, I = idx.range_search(qvecs, radius)
    print(f'IndexFlatL2 range search took {timer()-start:.2f}')
    return {'lims':lims, 'D':D, 'I':I}



def compute_neighborhood_pvalues(
        tcrs,
        fg_results,
        bg_ab_counts,
        num_bg_tcrs,
        radii = [24, 48, 72, 96],
        min_nbrs = 2, # tcr_clumping uses 1?
        pseudocount = 0.25,
        evalue_threshold = 1,
):
    ''' Compute pvalues and simple-bonferroni corrected "evalues"
    for observed foreground neighbor numbers

    Right now this is using poisson but we could shift to hypergeometric
    I think for the paired setting where the effective number of background comparisons
    is very large the two should give pretty similar results
    '''
    from scipy.stats import poisson

    # get background counts at radii
    maxdist = max(radii)
    assert bg_ab_counts.shape[1] == maxdist+1
    bg_counts = np.cumsum(bg_ab_counts, axis=1)[:, radii]

    # get foreground counts at radii
    num_fg_tcrs = tcrs.shape[0]
    lims = fg_results['lims']
    D = fg_results['D']
    assert num_fg_tcrs == lims.shape[0]-1
    fg_counts = np.zeros((num_fg_tcrs, len(radii)), dtype=int)
    for ii, r in enumerate(radii):
        fg_counts[:,ii] = np.add.reduceat((D<=r+.5), lims[:-1].astype(int))

    fg_counts -= 1 # exclude self nbrs

    assert fg_counts.shape == bg_counts.shape == (num_fg_tcrs, len(radii))

    # "rates" are the expected number of neighbors based on the background counts
    # we divide background counts by num_bg_tcrs**2 (since that's the effective number
    # of background paired comparisons) to get the probability of seeing a neighbor
    # at a given radius, then we multiply by num_fg_tcrs to get the expected number
    # of neighbors.
    # rates.shape: (num_fg_tcrs, len(radii))
    #
    rates = (np.maximum(bg_counts, pseudocount) *
             (num_fg_tcrs/(num_bg_tcrs*num_bg_tcrs)))

    dfl = []
    for ii, (counts,rates) in enumerate(zip(fg_counts, rates)):
        for jj, (count,rate) in enumerate(zip(counts,rates)):
            if count >= min_nbrs:
                pval = poisson.sf(count-1, rate)
                pval *= len(radii)*num_fg_tcrs
                if pval <= evalue_threshold:
                    dfl.append(dict(
                        evalue=pval,
                        tcr_index=ii,
                        radius=radii[jj],
                        num_nbrs=count,
                        expected_num_nbrs=rate,
                        bg_nbrs=rate*num_bg_tcrs*num_bg_tcrs/num_fg_tcrs,
                    ))
    results = pd.DataFrame(dfl)
    if results.shape[0]:
        results = results.join(tcrs, on='tcr_index').sort_values('evalue')
    return results



if __name__ == '__main__':


    tcrs = pd.read_table(args.tcrs)

    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'va', 'cdr3a', args.organism, 'A', j_column='ja')
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, 'vb', 'cdr3b', args.organism, 'B', j_column='jb')

    print(f'read and filtered {tcrs.shape[0]} foreground tcrs:')

    radii = [24, 48, 72, 96]

    bg_tcrs = make_background_tcrs(tcrs, args.organism, multiplier=args.bg_multiplier)

    bg_ab_counts = get_background_nbr_counts(
        tcrs, bg_tcrs, args.organism, aa_mds_dim=args.aa_mds_dim,
        maxdist = max(radii),
    )


    fg_results = get_foreground_nbr_counts(
        tcrs, args.organism, aa_mds_dim=args.aa_mds_dim, radius=max(radii)+0.5,
    )

    pvals = compute_neighborhood_pvalues(
        tcrs, fg_results, bg_ab_counts, bg_tcrs.shape[0], radii=radii,
        min_nbrs=2, pseudocount=0.25, evalue_threshold=1.,
    )


    pvals.to_csv(args.outfile, sep='\t', index=False)
