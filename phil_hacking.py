######################################################################################88
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
from os.path import exists
from glob import glob
from os import mkdir, system


# It may be helpful to take the sqrt of this matrix if we are
# going to use an L2 (Euclidean) distance in the embedding space...
# Also, turns out that when we take the sqrt it does satisfy the triangle
# inequality, which this "squared" version doesn't do.
#
TCRDIST_DM = np.maximum(0., np.minimum(4., 4-BLOSUM_62))

GAPCHAR = '-'

DATADIR = '/home/pbradley/gitrepos/immune_response_detection/data/' # change me
assert exists(DATADIR)

# so I can import this file in jupyter notebooks: 'if MAIN:' instead of 'if 1:'
MAIN = (__name__ == '__main__')

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


def parse_repertoire(
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


def resample_parsed_repertoire(
        parsed_df,
        num=None,
        match_j_families=False, # require ndn_part and j_part to have same j fam
        verbose=False,
):
    ''' Build a new random repertoire by mixing and matching cdr3 pieces
    from a parsed repertoire.

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


if 0: # setup for big calc
    # this block generates a file containing commands
    # those commands get distributed over the cluster
    #
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_hacking.py'

    radii = [3.5, 6.5, 12.5, 18.5, 24.5]
    num_repeats = 10

    fnames = glob('/home/pbradley/gitrepos/immune_response_detection/'
                  'data/phil/britanova/A*gz')
    print(len(fnames))

    runtag = 'run4' ; xargs = ' --max_tcrs 500000 ' # new 5th bg rep
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
                outfile_prefix = f'{rundir}{runtag}_{ftag}_{radius:.1f}_r{repeat}'
                cmd = (f'{PY} {EXE} {xargs} --filename {fname} --radius {radius} '
                       f' --outfile_prefix {outfile_prefix} '
                       f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
                out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)

    exit()



if MAIN: # try some range searching against various simple background repertoires
    # this block gets called by the commands in the listfile generated above
    import faiss
    import argparse
    import tcrdist

    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=float, required=True)
    parser.add_argument('--filename', required=True)
    parser.add_argument('--outfile_prefix', required=True)
    parser.add_argument('--max_tcrs', type=int)
    parser.add_argument('--aa_mds_dim', type=int, default=8)
    args = parser.parse_args()


    # load data from the Britanova aging study; I downloaded the files from:
    # https://zenodo.org/record/826447#.Y-7Ku-zMIWo
    #
    print('reading:', args.filename)

    tcrs = pd.read_table(args.filename)
    if args.max_tcrs is not None:
        tcrs = tcrs.head(args.max_tcrs)

    tcrs['v'] = tcrs.v+'*01'
    tcrs['j'] = tcrs.j+'*01'

    # remove singletons
    tcrs = tcrs[tcrs['count']>1]

    # filter bad genes/cdr3s
    v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
    tcrs = filter_out_bad_genes_and_cdr3s(
        tcrs, v_column, cdr3_column, organism, chain, j_column=j_column)
    print('num_tcrs:', tcrs.shape[0], args.filename)

    # encode the tcrs
    vecs = gapped_encode_tcr_chains(
        tcrs, organism, chain, args.aa_mds_dim, v_column=v_column,
        cdr3_column=cdr3_column).astype(np.float32)


    # parse repertoire, create background reps
    parsed_df = parse_repertoire(
        tcrs, organism, chain, v_column, j_column, cdr3_column)
    parsed_df_x1 = parse_repertoire(
        tcrs, organism, chain, v_column, j_column, cdr3_column, extend_align=1)

    tcrs['cdr3nt'] = tcrs.cdr3nt.str.lower()
    tcr_tuples = [(None,x) for x in tcrs['v j cdr3aa cdr3nt'.split()].itertuples(
        name=None, index=None)]
    junctions = tcrdist.tcr_sampler.parse_tcr_junctions(organism, tcr_tuples)

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


    for bgnum in reversed(range(5)):
        if bgnum==0:
            bg_tcrs = resample_parsed_repertoire(parsed_df)
        elif bgnum==1:
            bg_tcrs = resample_parsed_repertoire(parsed_df_x1)
        elif bgnum==2:
            bg_tcrs = resample_parsed_repertoire(
                parsed_df, match_j_families=True)
        elif bgnum==3:
            bg_tcrs = resample_parsed_repertoire(
                parsed_df_x1, match_j_families=True)
        elif bgnum==4:
            bg_tcr_tuples = tcrdist.tcr_sampler.resample_shuffled_tcr_chains(
                organism, tcrs.shape[0], chain, junctions)
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




