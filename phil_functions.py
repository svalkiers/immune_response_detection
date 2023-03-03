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


def update_hasher_aa_vectors_by_mds_tiling(dm, n_components, hasher):
    '''Try replacing hasher's aa_vectors_ with new ones created by tiling
    lower dimensional MDS vectors. Doesn't seem to be improve things much.
    '''
    assert dm.shape[0] == len(AALPHABET)
    vecs, stress = calc_mds_vecs(dm, n_components, return_stress=True)
    print(f'update_aa_vectors_by_mds_tiling: n_components= {n_components} '
          f'stress= {stress:.3f}')
    m = hasher.m
    for vec, aa in zip(vecs, AALPHABET):
        assert hasher.aa_vectors_[aa].shape == (m,) # sanity
        assert vec.shape == (n_components,)
        hasher.aa_vectors_[aa] = np.array([vec[i%n_components] for i in range(m)])
    return


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


def add_vdj_splits_info_to_junctions(
        junctions,
        min_d_length = 3, # otherwise it's hard to reliably place it
):
    ''' returns new copy of junctions that has more info on where the V/N/D/N/J
    regions start and stop, for use in a simulated background model

    last_v/first_d/last_d are -1 if not defined; first_j is len(cdr3nt)
    '''
    dfl = []
    for l in junctions.itertuples():
        seq = l.cdr3b_nucseq_src[:]
        nD = seq.count('D')
        if nD>0 and nD<min_d_length:
            seq = seq.replace('D','N')
        if seq[0] != 'V' or seq[-1] != 'J':
            print('whoah:', seq)

        # landmarks: last V, first D, last D, first J
        last_v, first_d, last_d, first_j = -1, -1, -1, len(seq)
        for ii,a in enumerate(seq):
            if a=='V':
                last_v = ii
            elif a=='D':
                first_d = ii
                break
        for ii,a in enumerate(reversed(seq)): # can't reverse an enumerate!
            if a=='J':
                first_j = len(seq)-1-ii
            elif a=='D':
                last_d = len(seq)-1-ii
                break

        has_d = (first_d != -1)

        v_nucseq = l.cdr3b_nucseq[:last_v+1]
        j_nucseq = l.cdr3b_nucseq[first_j:]
        if has_d:
            n1_nucseq = l.cdr3b_nucseq[last_v+1:first_d]
            d_nucseq = l.cdr3b_nucseq[first_d:last_d+1]
            n2_nucseq = l.cdr3b_nucseq[last_d+1:first_j]
        else:
            n1_nucseq = l.cdr3b_nucseq[last_v+1:first_j]
            d_nucseq = ''
            n2_nucseq = ''
        assert v_nucseq + n1_nucseq + d_nucseq + n2_nucseq + j_nucseq == l.cdr3b_nucseq

        dfl.append(dict(
            vdjseq = seq,
            last_v = last_v,
            first_d = first_d,
            last_d = last_d,
            first_j = first_j,
            has_d = has_d,
            v_nucseq = v_nucseq,
            n1_nucseq = n1_nucseq,
            d_nucseq = d_nucseq,
            n2_nucseq = n2_nucseq,
            j_nucseq = j_nucseq,
        ))
    df = pd.concat([junctions.reset_index(drop=True), pd.DataFrame(dfl)], axis=1)
    return df


def randomly_merge_two_nucseqs(a,b):
    ''' helper function for cominbining two N regions from different tcrs
    '''
    take_a_len = random.randint(0,1)
    minlen = min(len(a), len(b))
    split = random.randint(0,minlen)
    if take_a_len:
        from_b = b[len(b)-split:]
        from_a = a[:len(a)-split]
    else:
        from_a = a[:split]
        from_b = b[split:]
    return from_a + from_b


def resample_cdr3_nt_regions(junctions, match_lens=True):
    ''' background model based on mixing and matching cdr3 nucleotide pieces
    "junctions" comes from tcrdist.tcr_sampler.parse_tcr_junctions
    and then we have to call the add_vdj_splits_info_to_junctions function above
    '''
    import tcrdist
    all_tcrs = []
    for has_d in [True, False]:
        df = junctions[junctions.has_d==has_d]

        old_counts = Counter(len(x) for x in df.cdr3b)
        new_counts = Counter()
        tcrs = []
        N = df.shape[0]
        attempts = 0
        badlens = 0
        check_lengths = match_lens # turn it off if we waste too long
        while len(tcrs) < N:
            attempts += 1
            if attempts%10000==0:
                print('resample_junctions:', attempts, len(tcrs), badlens, has_d)
            vr = df.iloc[random.randint(0, N-1)]
            jr = df.iloc[random.randint(0, N-1)]

            if not has_d:
                # only choice is who contributes the N nucleotides
                assert not vr.has_d and not jr.has_d
                n_nucseq = randomly_merge_two_nucseqs(vr.n1_nucseq, jr.n1_nucseq)
                nucseq = (vr.v_nucseq + n_nucseq + jr.j_nucseq)

            else:
                dr = df.iloc[random.randint(0, N-1)]
                n1_nucseq = randomly_merge_two_nucseqs(vr.n1_nucseq, dr.n1_nucseq)
                n2_nucseq = randomly_merge_two_nucseqs(dr.n2_nucseq, jr.n2_nucseq)
                nucseq = (vr.v_nucseq + n1_nucseq + dr.d_nucseq + n2_nucseq +
                          jr.j_nucseq)
            if len(nucseq)%3 != 0:
                continue

            cdr3 = tcrdist.translation.get_translation(nucseq)
            assert len(cdr3) == len(nucseq)//3
            if check_lengths and new_counts[len(cdr3)] >= old_counts[len(cdr3)]:
                badlens += 1
                if badlens>len(tcrs): # too many failures
                    print('no more len-checking: too many badlens:', badlens, len(tcrs),
                          N, has_d)
                    check_lengths = False
                continue
            if '*' not in cdr3:
                new_counts[len(cdr3)] += 1
                tcrs.append((vr.vb, jr.jb, cdr3, nucseq))
        all_tcrs.extend(tcrs)
    assert len(all_tcrs) == junctions.shape[0]
    random.shuffle(all_tcrs)

    return all_tcrs



def parse_cdr3_aa_regions(
        df,
        organism,
        chain,
        v_column,
        j_column,
        cdr3_column,
        extend_align=0,
):
    ''' Returns new dataframe with info on where the V/J regions of the
    cdr3 amino acid sequence end/begin

    columns of new dataframe: ['v', 'j', 'cdr3', 'v_part', 'ndn_part', 'j_part']

    v_part is the part of the CDR3 that aligns with the V gene AA sequence
    j_part is the part of the CDR3 that aligns with the J gene AA sequence
    ndn_part is everything else (the middle; might be the empty string)

    '''
    all_genes_df = pd.read_table(DATADIR+'phil/combo_xcr.tsv')
    all_genes_df = all_genes_df[(all_genes_df.organism==organism)&
                                (all_genes_df.chain==chain)]

    all_genes_df['cdr3'] = all_genes_df.cdrs.str.split(';').str.get(-1).str.replace(
        '.','',regex=False)

    v_df = all_genes_df[all_genes_df.region=='V'].set_index('id')['cdr3']
    j_df = all_genes_df[all_genes_df.region=='J'].set_index('id')['cdr3']

    assert all(x in j_df.index for x in df[j_column])

    dfl = []
    for ind,v,j,cdr3 in df[[v_column,j_column,cdr3_column]].itertuples(name=None):
        v_cdr3 = v_df[v]
        j_cdr3 = j_df[j]

        v_idents = 0
        for a,b in zip(v_cdr3, cdr3):
            if a!=b:
                break
            else:
                v_idents += 1

        j_idents = 0
        for a,b in zip(reversed(j_cdr3), reversed(cdr3)):
            if a!=b:
                break
            else:
                j_idents += 1

        v_idents += extend_align
        j_idents += extend_align

        overlap = v_idents+j_idents - len(cdr3)
        if overlap>0:
            v_idents -= overlap//2
            j_idents -= (overlap - overlap//2)

        prefix = cdr3[:v_idents]
        middle = cdr3[v_idents:len(cdr3)-j_idents]
        suffix = cdr3[len(cdr3)-j_idents:]

        assert prefix+middle+suffix == cdr3
        if ind%25000==0:
            print(f'{prefix:7s} {middle:12s} {suffix:12s} {cdr3} {v_cdr3} {j_cdr3}',
                  ind, df.shape[0])

        dfl.append(dict(
            v=v,
            j=j,
            cdr3=cdr3,
            v_part=prefix,
            ndn_part=middle,
            j_part=suffix,
        ))

    parsed_df = pd.DataFrame(dfl)

    return parsed_df


def resample_cdr3_aa_regions(
        parsed_df,
        num=None,
        match_j_families=False, # require ndn_part and j_part to have same j fam
        verbose=False,
):
    ''' Build a new random repertoire by mixing and matching cdr3 pieces
    from a parsed repertoire.

    Pieces are defined at the amino acid level

    match the CDR3 length distributions
    '''
    if num is None:
        num = parsed_df.shape[0]
    # TODO: right now this only works if resampling the same number of tcrs,
    #   because of cdr3-length-distribution-matching
    assert num == parsed_df.shape[0]

    old_counts = Counter(parsed_df.cdr3.str.len())
    new_counts = Counter() # the new cdr3 len counts

    dfl = []
    skipcount = 0 # diagnostics

    while len(dfl)<num:
        replace = num>parsed_df.shape[0]
        v_parts = parsed_df.sample(num, replace=replace)
        d_parts = parsed_df.sample(num, replace=replace)
        j_parts = parsed_df.sample(num, replace=replace)

        for vrow, drow, jrow in zip(v_parts.itertuples(),
                                    d_parts.itertuples(),
                                    j_parts.itertuples()):
            counter= vrow.Index
            if verbose and counter%10000==0:
                print(counter, len(dfl), skipcount)
            v = vrow.v
            j = jrow.j
            if match_j_families and j[2] == 'B':
                assert j.startswith('TRBJ') and j[5] == '-'
                j_jfam = int(jrow.j[4])
                d_jfam = int(drow.j[4])
                assert j_jfam in [1,2] and d_jfam in [1,2]
                if j_jfam != d_jfam:
                    skipcount += 1
                    continue

            cdr3 = vrow.v_part + drow.ndn_part + jrow.j_part
            l = len(cdr3)
            if new_counts[l] < old_counts[l]:
                dfl.append(dict(v=v, j=j, cdr3=cdr3))
                new_counts[l] += 1

                if len(dfl) >= num:
                    break
            else:
                skipcount += 1

    assert len(dfl) == num
    if verbose:
        print('final skipcount:', skipcount)
    return pd.DataFrame(dfl)

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

def resample_background_tcrs_v4(organism, chain, junctions):
    from tcrdist.tcr_sampler import parse_tcr_junctions, resample_shuffled_tcr_chains

    multiplier = 3 # so we have enough to match distributions
    bg_tcr_tuples, src_junction_indices = resample_shuffled_tcr_chains(
        organism, multiplier * junctions.shape[0], chain, junctions,
        return_src_junction_indices=True,
    )

    resamples = []
    for tcr, inds in zip(bg_tcr_tuples, src_junction_indices):
        if len(resamples)%50000==0:
            print('resample_background_tcrs_v4: build nucseq_srclist', len(resamples),
                  len(bg_tcr_tuples))
        v,j,cdr3aa,cdr3nt = tcr
        vjunc = junctions.iloc[inds[0]]
        jjunc = junctions.iloc[inds[1]]
        nucseq_src = (vjunc.cdr3b_nucseq_src[:inds[2]] +
                      jjunc.cdr3b_nucseq_src[inds[2]:])
        assert len(tcr[3]) == len(nucseq_src)
        #dfl.append(dict(v=v, j=j, cdr3aa=cdr3aa, cdr3nt=cdr3nt, nucseq_src=nucseq_src))
        resamples.append((v, j, cdr3aa, cdr3nt, nucseq_src))

    fg_nucseq_srcs = list(junctions.cdr3b_nucseq_src)

    # try to match lengths first
    fg_lencounts = Counter(len(x.cdr3b) for x in junctions.itertuples())

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

    assert len(good_resamples)
    # now try to match the N insertion distributions, while preserving the
    # length distributions
    fg_ncounts = Counter(x.count('N') for x in fg_nucseq_srcs)
    bg_ncounts = Counter(x[4].count('N') for x in good_resamples)
    all_ncounts = Counter(x[4].count('N') for x in resamples)

    tries = 0
    too_many_tries = 10*junctions.shape[0]

    for ii in range(len(bad_resamples)):
        tries += 1
        iilen = len(bad_resamples[ii][2])
        if bg_lencounts[iilen]==0:
            continue
        iinc = bad_resamples[ii][4].count('N')
        if iinc in [0,1] and fg_ncounts[iinc] > bg_ncounts[iinc]: # most biased
            # find good_resamples with same len, elevated nc
            while True:
                tries += 1
                jj = np.random.randint(0,len(good_resamples))
                jjlen = len(good_resamples[jj][2])
                if jjlen != iilen:
                    continue
                jjnc = good_resamples[jj][4].count('N')
                if bg_ncounts[jjnc] > fg_ncounts[jjnc]:
                    break
                if tries > too_many_tries:
                    break
            if tries > too_many_tries:
                print('WARNING too_many_tries2:', tries)
                break
            #print('swap:', iinc, jjnc, iilen, fg_ncounts[iinc]-bg_ncounts[iinc],
            #      tries)
            tmp = good_resamples[jj]
            good_resamples[jj] = bad_resamples[ii]
            bad_resamples[ii] = tmp
            bg_ncounts[iinc] += 1
            bg_ncounts[jjnc] -= 1


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

    return [x[:4] for x in good_resamples]
    #return fg_ncounts, bg_ncounts, all_ncounts
