######################################################################################88
#
# this has function definitions etc for importing in other scripts
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
import random
import faiss
import itertools as it

# It may be helpful to take the sqrt of this matrix if we are
# going to use an L2 (Euclidean) distance in the embedding space...
# Also, turns out that when we take the sqrt it does satisfy the triangle
# inequality, which this "squared" version doesn't do.
#
TCRDIST_DM = np.maximum(0., np.minimum(4., 4-BLOSUM_62))

GAPCHAR = '-'

DATADIR = '/home/pbradley/gitrepos/immune_response_detection/data/' # change me
assert exists(DATADIR)

def calc_mds_vecs(dm, n_components, return_stress = False):
    'Helper function to run MDS'
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=11,
              normalized_stress=False)
    vecs = mds.fit_transform(dm)
    if return_stress:
        return vecs, mds.stress_
    else:
        return vecs


def trim_and_gap_cdr3(cdr3, num_pos, n_trim=3, c_trim=2):
    ''' Convert a variable length cdr3 to a fixed-length sequence in a way
    that is consistent with tcrdist scoring, by trimming the ends and
    inserting gaps at a fixed position

    If the cdr3 is longer than num_pos + n_trim + c_trim, some residues will be dropped
    '''
    gappos = min(6, 3+(len(cdr3)-5)//2) - n_trim
    r = -c_trim if c_trim>0 else len(cdr3)
    seq = cdr3[n_trim:r]
    afterlen = min(num_pos-gappos, len(seq)-gappos)
    numgaps = max(0, num_pos-len(seq))
    fullseq = seq[:gappos] + GAPCHAR*numgaps + seq[-afterlen:]
    assert len(fullseq) == num_pos
    return fullseq

def encode_sequence(seq, aa_vectors):
    ''' Convert a sequence to a vector by lining up the aa_vectors

    length of the vector will be dim * len(seq), where dim is the dimension of the
    embedding given by aa_vectors
    '''
    dim = aa_vectors['A'].shape[0]
    vec = np.zeros((len(seq)*dim,))
    for i,aa in enumerate(seq):
        vec[i*dim:(i+1)*dim] = aa_vectors[aa]
    return vec

def gapped_encode_cdr3(cdr3, aa_vectors, num_pos, n_trim=3, c_trim=2):
    ''' Convert a cdr3 of variable length to a fixed-length vector
    by trimming/gapping and then lining up the aa_vectors

    length of the vector will be dim * num_pos, where dim is the dimension of the
    embedding given by aa_vectors
    '''
    return encode_sequence(trim_and_gap_cdr3(cdr3, num_pos, n_trim, c_trim),
                           aa_vectors)


def calc_tcrdist_aa_vectors(dim, SQRT=True, verbose=False):
    ''' Embed tcrdist distance matrix to Euclidean space
    '''
    dm = np.zeros((21,21))
    dm[:20,:20] = TCRDIST_DM
    dm[:20,20] = 4.
    dm[20,:20] = 4.
    if SQRT:
        dm = np.sqrt(dm) ## NOTE
    vecs, stress = calc_mds_vecs(dm, dim, return_stress=True)
    #print('vecs mean:', np.mean(vecs, axis=0)) #looks like they are already zeroed
    vecs -= np.mean(vecs, axis=0) # I think this is unnecessary, but for sanity...
    if verbose:
        print(f'encoding tcrdist aa+gap matrix, dim= {dim} stress= {stress}')
    aa_vectors = {aa:v for aa,v in zip(AALPHABET+GAPCHAR, vecs)}
    return aa_vectors



def gapped_encode_cdr3s(cdr3s, dim, num_pos, SQRT=True, n_trim=3, c_trim=2):
    ''' Convert a list/Repertoire of cdr3s to fixed-length vectors
    Uses the gapped_encode_cdr3 function above, and an aa embedding from MDS
    of the tcrdist distance matrix.
    '''
    aa_vectors = calc_tcrdist_aa_vectors(dim, SQRT=SQRT, verbose=True)
    return np.array([gapped_encode_cdr3(cdr3, aa_vectors, num_pos, n_trim, c_trim)
                     for cdr3 in cdr3s])


def setup_gene_cdr_strings(organism, chain):
    ''' returns dict mapping vgene names to concatenated cdr1-cdr2-cdr2.5 strings
    columns without any sequence variation (e.g. all gaps) are removed
    '''
    # remove tcrdist dependency here
    all_genes_df = pd.read_table(DATADIR+'phil/combo_xcr.tsv')
    all_genes_df = all_genes_df[(all_genes_df.organism==organism)&
                                (all_genes_df.chain==chain)&
                                (all_genes_df.region=='V')]
    assert all_genes_df.id.value_counts().max()==1
    all_genes_df.set_index('id', inplace=True)
    all_genes_df['cdrs'] = all_genes_df.cdrs.str.split(';')

    assert chain in ['A','B']
    vgenes = list(all_genes_df.index)
    gene_cdr_strings = {x:'' for x in vgenes}

    oldgap = '.' # gap character in the all_genes dict
    for icdr in range(3):
        cdrs = all_genes_df.cdrs.str.get(icdr).str.replace(oldgap,GAPCHAR,regex=False)
        L = len(cdrs[0])
        for i in reversed(range(L)):
            col = set(x[i] for x in cdrs)
            if len(col) == 1:# no variation
                #print('drop fixed col:', col, organism, chain, icdr, i)
                cdrs = [x[:i]+x[i+1:] for x in cdrs]
        for g, cdr in zip(vgenes, cdrs):
            gene_cdr_strings[g] += cdr
    return gene_cdr_strings



def gapped_encode_tcr_chains(
        tcrs,
        organism,
        chain,
        aa_mds_dim,
        v_column = 'v_call',
        cdr3_column = 'junction_aa',
        num_pos_cdr3=16,
        cdr3_weight=3.0,
):
    ''' tcrs is a DataFrame with V and CDR3 information in the named columns
    organism is 'human' or 'mouse'
    chain is 'A' or 'B'
    '''
    assert organism in ['human','mouse']
    assert chain in ['A','B']
    aa_vectors = calc_tcrdist_aa_vectors(aa_mds_dim, SQRT=True, verbose=True)
    gene_cdr_strings = setup_gene_cdr_strings(organism, chain)
    num_pos_other_cdrs = len(next(iter(gene_cdr_strings.values())))
    assert all(len(x)==num_pos_other_cdrs for x in gene_cdr_strings.values())

    vec_len = aa_mds_dim * (num_pos_other_cdrs + num_pos_cdr3)
    print('gapped_encode_tcr_chains: aa_mds_dim=', aa_mds_dim,
          'num_pos_other_cdrs=', num_pos_other_cdrs,
          'num_pos_cdr3=', num_pos_cdr3, 'vec_len=', vec_len)

    vecs = []
    for v, cdr3 in zip(tcrs[v_column], tcrs[cdr3_column]):
        v_vec = encode_sequence(gene_cdr_strings[v], aa_vectors)
        cdr3_vec = np.sqrt(cdr3_weight) * gapped_encode_cdr3(
            cdr3, aa_vectors, num_pos_cdr3)
        vecs.append(np.concatenate([v_vec, cdr3_vec]))
    vecs = np.array(vecs)
    assert vecs.shape == (tcrs.shape[0], vec_len)
    return vecs






def get_nonself_nbrs_from_distances(D_in, num_nbrs):
    'Returns the set() of all neighbor pairs, not including self in nbr list'
    D = D_in.copy()
    N = D.shape[0]
    inds = np.arange(N)
    D[inds,inds] = 1e6
    nbrs = np.argpartition(D, num_nbrs-1)[:,:num_nbrs]
    nbrs_set = set((i,n) for i,i_nbrs in enumerate(nbrs) for n in i_nbrs)
    return nbrs_set

def get_nonself_nbrs_from_distances_by_radius(D_in, radius):
    'Returns the set() of all neighbor pairs (i,j), not including i>=j pairs'
    iis, jjs = np.nonzero(D_in<radius)
    nbrs_set = set( (i,j) for i,j in zip(iis,jjs) if i<j)
    return nbrs_set


def compute_gapped_seqs_dists(seqs, force_metric_dim=None):
    ''' For sanity checking that our gapping procedure gives distances that
    are consistent with tcrdist
    '''
    assert all(len(x) == len(seqs[0]) for x in seqs) # fixed-len

    dm = np.zeros((21,21))
    dm[:20,:20] = TCRDIST_DM
    dm[:20,20] = 4.
    dm[20,:20] = 4.

    if force_metric_dim is not None:
        vecs, stress = calc_mds_vecs(dm, force_metric_dim, return_stress=True)
        new_dm = squareform(pdist(vecs))
        avg_error = np.mean((dm-new_dm)**2)
        print(f'compute_gapped_seqs_dists force_metric_dim= {force_metric_dim} '
              f'stress= {stress:.6f} avg_error: {avg_error:.6f}')
        dm = new_dm

    aa2index = {aa:i for i,aa in enumerate(AALPHABET+GAPCHAR)}

    D = np.zeros((len(seqs), len(seqs)))

    for i,a in enumerate(seqs):
        for j,b in enumerate(seqs):
            if i<j:
                dist = sum(dm[aa2index[x],aa2index[y]]
                           for x,y in zip(a,b))
                D[i,j] = dist
                D[j,i] = dist
    return D


def filter_out_bad_genes_and_cdr3s(
        df,
        v_column,
        cdr3_column,
        organism,
        chain,
        min_cdr3_len = 6,
        j_column=None,
):
    ''' returns filtered copy of df

    removes tcrs with

    * unrecognized V gene names (and J genes, if j_column != None)
    * V genes whos cdr1/cdr2/cdr2.5 contain '*' (probably pseudogenes?)
    * CDR3s with non-AA characters or shorter than 6
    '''
    all_genes_df = pd.read_table(DATADIR+'phil/combo_xcr.tsv')
    all_genes_df = all_genes_df[(all_genes_df.organism==organism)&
                                (all_genes_df.chain==chain)]
    # drop cdr3, since a '*' there might be trimmed back so it's OK...
    all_genes_df['cdrs'] = all_genes_df.cdrs.str.split(';').str.slice(0,3).str.join(';')

    known_v_genes = set(all_genes_df[all_genes_df.region=='V'].id)
    bad_v_genes = set(x.id for x in all_genes_df.itertuples()
                      if x.region == 'V' and '*' in x.cdrs)
    print('bad_v_genes:', len(bad_v_genes), bad_v_genes)

    good_cdr3s_mask = np.array(
        [len(cdr3)>=min_cdr3_len and all(aa in AALPHABET for aa in cdr3)
         for cdr3 in df[cdr3_column]])
    print('bad_cdr3s in df:', (~good_cdr3s_mask).sum())

    bad_v_genes_mask = df[v_column].isin(bad_v_genes)
    print('bad_v_genes in df:', bad_v_genes_mask.sum(),
          df[bad_v_genes_mask][v_column].unique())

    unknown_genes_mask = ~df[v_column].isin(known_v_genes)
    print('unknown_genes in df:', unknown_genes_mask.sum(),
          df[unknown_genes_mask][v_column].unique())
    if j_column is not None:
        known_j_genes = set(all_genes_df[all_genes_df.region=='J'].id)
        unknown_j_genes_mask = ~df[j_column].isin(known_j_genes)
        print('unknown_j_genes in df:', unknown_j_genes_mask.sum(),
              df[unknown_j_genes_mask][j_column].unique())
        unknown_genes_mask |= unknown_j_genes_mask


    return df[good_cdr3s_mask & (~bad_v_genes_mask) & (~unknown_genes_mask)].copy()



def compute_nbr_count_pvalues(
        fg_counts,
        bg_counts,
        num_bg_tcrs,
        min_fg_nbrs = 2,
        min_fg_bg_nbr_ratio=2.0,
        max_fg_bg_nbr_ratio=100.0, # used if bg_nbrs==0 (doesnt affect pval calc)
        min_pvalue = 1e-300,
        target_bg_nbrs=None, # for the 'rescaled' pvalue
):
    ''' Compute hypergeometric pvalues from foreground and background neighbor counts
    '''
    from scipy.stats import hypergeom
    num_fg_tcrs = fg_counts.shape[0]
    assert bg_counts.shape == (num_fg_tcrs,)

    dfl = []
    for ind, (fg_nbrs, bg_nbrs) in enumerate(zip(fg_counts, bg_counts)):
        if ind%50000==0:
            print('compute_nbr_count_pvalues:', ind, num_fg_tcrs)
        expected_nbrs = bg_nbrs * num_fg_tcrs / num_bg_tcrs
        if bg_nbrs==0:
            if fg_nbrs==0:
                fg_bg_nbr_ratio = 1.
            else:
                fg_bg_nbr_ratio = max_fg_bg_nbr_ratio
        else:
            fg_bg_nbr_ratio = fg_nbrs / expected_nbrs
        if fg_nbrs < min_fg_nbrs or fg_bg_nbr_ratio < min_fg_bg_nbr_ratio:
            pval = 1.
            rescaled_pval = 1.
        else:
            pval = hypergeom.sf(fg_nbrs-1, num_fg_tcrs+num_bg_tcrs-1, fg_nbrs+bg_nbrs,
                                num_fg_tcrs-1)
            pval = max(min_pvalue, pval)
            rescaled_pval = pval
            if target_bg_nbrs is not None and target_bg_nbrs < bg_nbrs:
                rescale = target_bg_nbrs / bg_nbrs
                new_num_bg_tcrs = np.ceil(rescale * num_bg_tcrs)
                rescaled_pval = hypergeom.sf(
                    fg_nbrs-1, num_fg_tcrs + new_num_bg_tcrs-1,
                    fg_nbrs+target_bg_nbrs, num_fg_tcrs-1)
        dfl.append(dict(
            tcr_index = ind,
            pvalue = pval,
            evalue = num_fg_tcrs * pval,
            rescaled_pvalue = rescaled_pval,
            rescaled_evalue = num_fg_tcrs * rescaled_pval,
            fg_nbrs = fg_nbrs,
            bg_nbrs = bg_nbrs,
            fg_bg_nbr_ratio = fg_bg_nbr_ratio,
            expected_nbrs = expected_nbrs,
            num_fg_tcrs = num_fg_tcrs,
            num_bg_tcrs = num_bg_tcrs,
        ))
    return pd.DataFrame(dfl)

def read_britanova_tcrs(filename, max_tcrs=None, min_count=2):
    # load data from the Britanova aging study; I downloaded the files from:
    # https://zenodo.org/record/826447#.Y-7Ku-zMIWo
    tcrs = pd.read_table(filename)
    if max_tcrs is not None:
        tcrs = tcrs.head(max_tcrs)

    acount = tcrs.v.str.contains('*', regex=False).sum()
    if acount==0:
        tcrs['v'] = tcrs.v+'*01'
        tcrs['j'] = tcrs.j+'*01'
    else:
        assert acount==tcrs.shape[0]
    tcrs['cdr3nt'] = tcrs.cdr3nt.str.lower()

    # remove singletons
    tcrs = tcrs[tcrs['count']>=min_count]

    # filter bad genes/cdr3s
    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, v_column, cdr3_column, organism, chain, j_column=j_column)
    print('num_tcrs:', tcrs.shape[0], filename)
    return tcrs


def parse_junctions_for_background_resampling(
        tcrs,
        organism,
        chain, # 'A' or 'B'
        v_column,
        j_column,
        cdr3aa_column,
        cdr3nt_column,
):
    ''' setup for the "resample_background_tcrs_v4" function by parsing
    the V(D)J junctions in the foreground tcr set

    returns a dataframe with info
    '''
    from tcrdist.tcr_sampler import parse_tcr_junctions

    # tcrdist parsing function expects paired tcrs as list of tuples of tuples
    cols = [v_column, j_column, cdr3aa_column, cdr3nt_column]
    tcr_tuples = tcrs[cols].itertuples(name=None, index=None)
    if chain == 'A':
        tcr_tuples = zip(tcr_tuples, it.repeat(None))
    else:
        tcr_tuples = zip(it.repeat(None), tcr_tuples)

    junctions = parse_tcr_junctions(organism, list(tcr_tuples))
    #junctions = add_vdj_splits_info_to_junctions(junctions)
    return junctions


def resample_background_tcrs_v4(
        organism, # human or mouse
        chain, # A or B
        junctions, # junctions info dataframe created by the function above this one
        preserve_vj_pairings = False, # consider setting True for alpha chain
        return_nucseq_srcs = False, # for debugging
):
    ''' Resample shuffled background sequences, number will be equal to size of
    foreground repertore, ie junctions.shape[0]

    junctions is a dataframe with information about the V(D)J junctions in the
    foreground tcrs. Created by the function above this one,
        "parse_junctions_for_background_resampling"

    returns a list of tuples [(v,j,cdr3aa,cdr3nt), ...] of length = junctions.shape[0]

    '''
    from tcrdist.tcr_sampler import resample_shuffled_tcr_chains

    assert chain in ['A','B']

    multiplier = 3 # so we have enough to match distributions
    bg_tcr_tuples, src_junction_indices = resample_shuffled_tcr_chains(
        organism, multiplier * junctions.shape[0], chain, junctions,
        preserve_vj_pairings = preserve_vj_pairings,
        return_src_junction_indices=True,
    )

    fg_nucseq_srcs = list(junctions[f'cdr3{chain.lower()}_nucseq_src'])

    resamples = []
    for tcr, inds in zip(bg_tcr_tuples, src_junction_indices):
        if len(resamples)%500000==0:
            print('resample_background_tcrs_v4: build nucseq_srclist', len(resamples),
                  len(bg_tcr_tuples))
        v,j,cdr3aa,cdr3nt = tcr
        v_nucseq = fg_nucseq_srcs[inds[0]]
        j_nucseq = fg_nucseq_srcs[inds[1]]
        nucseq_src = v_nucseq[:inds[2]] + j_nucseq[inds[2]:]
        assert len(tcr[3]) == len(nucseq_src)
        resamples.append((v, j, cdr3aa, cdr3nt, nucseq_src))


    # try to match lengths first
    fg_lencounts = Counter(len(x) for x in junctions['cdr3'+chain.lower()])

    N = junctions.shape[0]
    good_resamples = resamples[:N]
    bad_resamples = resamples[N:]

    bg_lencounts = Counter(len(x[2]) for x in good_resamples)
    all_lencounts = Counter(len(x[2]) for x in resamples)
    if not all(all_lencounts[x]>=fg_lencounts[x] for x in fg_lencounts):
        print('dont have enough of all lens')

    tries = 0
    too_many_tries = 10*junctions.shape[0]

    while True:
        tries += 1
        # pick a good tcr with a bad length and a bad tcr with a good length, swap them
        ii = np.random.randint(0,len(bad_resamples))
        iilen = len(bad_resamples[ii][2])
        if bg_lencounts[iilen] < fg_lencounts[iilen]: # too few of len=iilen
            tries = 0
            while True:
                tries += 1
                jj = np.random.randint(0,len(good_resamples))
                jjlen = len(good_resamples[jj][2])
                if bg_lencounts[jjlen] > fg_lencounts[jjlen] or tries>too_many_tries:
                    break

            if tries>too_many_tries:
                print('WARNING too_many_tries1:', tries)
                break
            # swap!
            dev = sum(abs(fg_lencounts[x]-bg_lencounts[x]) for x in fg_lencounts)
            #print(f'swap: {dev} {iilen} {jjlen} {tries} {too_many_tries}')
            tmp = good_resamples[jj]
            good_resamples[jj] = bad_resamples[ii]
            bad_resamples[ii] = tmp
            bg_lencounts[iilen] += 1
            bg_lencounts[jjlen] -= 1

            # are we done? if so, break out
            if all((fg_lencounts[x]<=bg_lencounts[x] or
                    bg_lencounts[x]==all_lencounts[x]) for x in fg_lencounts):
                break
            else:
                pass
                # print('devs:', end=' ')
                # for ii in range(100):
                #     if fg_lencounts[ii] != bg_lencounts[ii]:
                #         print(ii, fg_lencounts[ii]-bg_lencounts[ii], end=' ')
                # print()

    assert len(good_resamples) == N

    fg_ncounts = Counter(x.count('N') for x in fg_nucseq_srcs)
    bg_ncounts = Counter(x[4].count('N') for x in good_resamples)

    if chain == 'B':
        desirable_Ncounts = [0,1]
    else:
        desirable_Ncounts = [x for x in range(10)
                             if bg_ncounts[x] < 0.9 * fg_ncounts[x]]
    print('desirable_Ncounts:', desirable_Ncounts)
    bad_resamples = [x for x in bad_resamples if x[4].count('N') in desirable_Ncounts]

    # now try to match the N insertion distributions, while preserving the
    # length distributions
    bad_ncounts = Counter(x[4].count('N') for x in bad_resamples)
    all_ncounts = Counter(x[4].count('N') for x in resamples)

    tries = 0
    too_many_tries = 10*junctions.shape[0]
    too_many_inner_tries = 1000

    for num in desirable_Ncounts:
        print('Ns:', num, 'fg_ncounts:', fg_ncounts[num],
              'bg_ncounts:', bg_ncounts[num], 'bad_ncounts:', bad_ncounts[num],
              'sum:', bg_ncounts[num] + bad_ncounts[num])
        if fg_ncounts[num] > all_ncounts[num]:
            print('resample_background_tcrs_v4 dont have enough Ns:',
                  num, fg_ncounts[num], '>', all_ncounts[num])

    for ii in range(len(bad_resamples)):
        tries += 1
        if tries > too_many_tries:
            print('WARNING too_many_tries2:', tries)
            break
        iilen = len(bad_resamples[ii][2])
        if bg_lencounts[iilen]==0:
            continue
        iinc = bad_resamples[ii][4].count('N')
        assert iinc in desirable_Ncounts # now by construction
        if iinc in desirable_Ncounts and fg_ncounts[iinc] > bg_ncounts[iinc]:
            # find good_resamples with same len, elevated nc
            inner_tries = 0
            while True:
                inner_tries += 1
                jj = np.random.randint(0,len(good_resamples))
                jjlen = len(good_resamples[jj][2])
                if jjlen != iilen:
                    continue
                jjnc = good_resamples[jj][4].count('N')
                if bg_ncounts[jjnc] > fg_ncounts[jjnc]:
                    break
                if inner_tries > too_many_inner_tries:
                    break
            if inner_tries > too_many_inner_tries:
                tries += inner_tries//10
                continue

            #print('swap:', iinc, jjnc, iilen, fg_ncounts[iinc]-bg_ncounts[iinc],
            #      tries)
            tmp = good_resamples[jj]
            good_resamples[jj] = bad_resamples[ii]
            bad_resamples[ii] = tmp
            bg_ncounts[iinc] += 1
            bg_ncounts[jjnc] -= 1
            bad_ncounts[iinc] -= 1
            bad_ncounts[jjnc] += 1
            if all(bad_ncounts[x] == 0 for x in desirable_Ncounts):
                print('ran out of desirable_Ncounts:', desirable_Ncounts)
                break


    print('final Ndevs:', end=' ')
    for ii in range(100):
        if fg_ncounts[ii] != bg_ncounts[ii]:
            print(ii, fg_ncounts[ii]-bg_ncounts[ii], end=' ')
    print()

    print('final Ldevs:', end=' ')
    for ii in range(100):
        if fg_lencounts[ii] != bg_lencounts[ii]:
            print(ii, fg_lencounts[ii]-bg_lencounts[ii], end=' ')
    print()

    good_tcrs = [x[:4] for x in good_resamples]

    if return_nucseq_srcs:
        good_nucseq_srcs = [x[4] for x in good_resamples]
        return good_tcrs, good_nucseq_srcs
    else:
        return good_tcrs



def sample_igor_tcrs(
        num,
        v_column='v',
        j_column='j',
        cdr3aa_column='cdr3aa',
        cdr3nt_column='cdr3nt',
        random_state=None,
):
    tag = ('_1e5' if num <= 1e5 else
           '_5e5' if num <= 5e5 else
           '_1e6' if num <= 1e6 else
           '_2e6' if num <= 2e6 else
           '')
    fname = ('/home/pbradley/gitrepos/immune_response_detection/data/phil/'
             f'big_background_filt{tag}.tsv')
    print('reading:', fname)
    tcrs = pd.read_table(fname)
    assert tcrs.shape[0] >= num

    return tcrs.sample(num, random_state=random_state).rename(columns={
        'cdr3nt':cdr3nt_column,
        'cdr3aa':cdr3aa_column,
        'v':v_column,
        'j':j_column,
    })

def compute_background_single_tcrdist_distributions(
        fg_vecs,
        bg_vecs,
        maxdist,
        rowstep = 5000,
        colstep = 50000,
):
    dim = fg_vecs.shape[1]
    assert dim == bg_vecs.shape[1]
    num_fg = fg_vecs.shape[0]
    num_bg = bg_vecs.shape[0]
    maxdist = int(maxdist+0.1) # confirm int
    print('compute_background_single_tcrdist_distributions: '
          f'dim= {dim} maxdist= {maxdist}')
    #maxdist_float = maxdist + 0.5
    dist_counts = np.zeros((num_fg, maxdist+1), dtype=int)

    nrows = (num_fg-1)//rowstep + 1
    ncols = (num_bg-1)//colstep + 1

    # initialize distance storage
    dists = np.zeros((rowstep*colstep,), dtype=np.float32) # 1D array

    start0 = timer()
    for ii in range(nrows):
        ii_start = ii*rowstep
        for jj in range(ncols):
            jj_start = jj*colstep
            xq = fg_vecs[ii_start:ii_start+rowstep] # faiss terminology here
            xb = bg_vecs[jj_start:jj_start+colstep]
            nq = xq.shape[0] # Num Query
            nb = xb.shape[0] # Num dataBase (?)

            start = timer()
            faiss.pairwise_L2sqr(dim, nq, faiss.swig_ptr(xq), nb, faiss.swig_ptr(xb),
                                 faiss.swig_ptr(dists))
            disttime = timer()-start
            #print(disttime)
            start += disttime

            # now fill counts
            for iq in range(nq):
                dist_counts[ii_start+iq,:] += np.histogram(
                    dists[iq*nb:(iq+1)*nb], bins = maxdist+1,
                    range = (-0.5, maxdist+0.5))[0]

            counttime = timer()-start

            #print(f'ij {ii} {jj} {disttime:.2f} {counttime:.2f} {timer()-start0:.2f}')

    return dist_counts

def compute_background_paired_tcrdist_distributions(
        fg_avecs,
        fg_bvecs,
        bg_avecs,
        bg_bvecs,
        maxdist,
        rowstep = 5000,
        colstep = 50000,
):
    ''' Compute the background paired tcrdist distribution by taking the
    convolution of the alpha and beta single-chain tcrdist distributions
    '''

    num_fg_tcrs = fg_avecs.shape[0]
    num_bg_tcrs = bg_avecs.shape[0]
    assert num_fg_tcrs == fg_bvecs.shape[0]
    assert num_bg_tcrs == bg_bvecs.shape[0]

    acounts = compute_background_single_tcrdist_distributions(
        fg_avecs, bg_avecs, maxdist, rowstep=rowstep, colstep=colstep)

    bcounts = compute_background_single_tcrdist_distributions(
        fg_bvecs, bg_bvecs, maxdist, rowstep=rowstep, colstep=colstep)

    abcounts = np.zeros((num_fg_tcrs, maxdist+1), dtype=int)

    assert acounts.shape == bcounts.shape == abcounts.shape

    for d in range(maxdist+1):
        for adist in range(d+1):
            abcounts[:,d] += acounts[:,adist] * bcounts[:,d-adist]

    return abcounts

