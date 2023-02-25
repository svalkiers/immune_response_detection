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
import sys
from os.path import exists
from glob import glob
from os import mkdir, system
import random

from phil_functions import *

#MAIN = __name__ == '__main__'

## testing new functions:

def read_yfv_tcrs(fname, min_count=2):
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

#####################################



if 0: # read, parse, and subset the YFV d15 tcrs
    min_count = 2
    yfvdir = '/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
    files = glob(yfvdir+'??_15_F?_.txt')
    assert len(files) == 7

    for fname in files:
        tcrs = read_yfv_tcrs(fname)
        outfile = f'{fname[:-4]}_min_count_{min_count}.txt'
        tcrs.to_csv(outfile, sep='\t', index=False)
        print('made:', outfile)

    exit()

if 0: # testing new background model
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
    import faiss

    runtag = 'run12'
    #runtag = 'run11'
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

    fg_files = sorted(glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
                           '*min_count_2.txt.gz'))
    assert len(fg_files) == 7
    #fg_files = fg_files[2:3]

    nrows, ncols = 2, 4
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

        plt.scatter(fg_nndists[xnbrs2_mask], bg_nndists[xnbrs2_mask], s=5, alpha=0.5)
        plt.scatter(fg_nndists[xnbrs1_mask], bg_nndists[xnbrs1_mask], s=5, alpha=0.5)

        xinds = np.nonzero(np.array(xmask))[0]
        plt.scatter(fg_nndists[xinds], bg_nndists[xinds], s=5, alpha=1)



    plt.tight_layout()

    pngfile = ('/home/pbradley/csdat/raptcr/'
               f'yfv_{runtag}_nndists_F{len(fg_files)}_bg{bgnum}.png')
    plt.savefig(pngfile, dpi=300)
    print('made:', pngfile)

    exit()

if 1: # look at nbr counts in YFV data
    import tcrdist
    import networkx as nx
    from scipy.spatial.distance import squareform, pdist
    import faiss

    aa_mds_dim = 8
    fg_files = sorted(glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
                           '*min_count_2.txt.gz'))
    assert len(fg_files) == 7


    fg_files = fg_files[3:4]

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


    radius = 12.5
    bgnum = 4
    max_evalue = 0.1
    num_lines = 50
    target_bg_nbrs = None

    nplots = 3*len(fg_files)
    nrows = max(1, int(0.8*np.sqrt(nplots)))
    ncols = (nplots-1)//nrows + 1
    print(nrows, ncols, fg_files)
    plt.figure(figsize=(ncols*3, nrows*3))
    pngfile = ('/home/pbradley/csdat/raptcr/'
               f'yfv_run9_pvals_F{len(fg_files)}_{radius:.1f}_bg{bgnum}_'
               f'tbn{target_bg_nbrs}.png')

    run9prefix = '/home/pbradley/csdat/raptcr/slurm/run9/run9'
    run10prefix = '/home/pbradley/csdat/raptcr/slurm/run10/run10'
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

        my_counts_files = glob(f'{run9prefix}_{fg_tag}_{radius:.1f}_*npy')
        print(fg_tag, 'numfiles:', len(my_counts_files))
        if len(my_counts_files)<70:
            continue
        assert len(my_counts_files) == 70

        # read fg counts
        fg_counts = np.load(f'{run9prefix}_{fg_tag}_{radius:.1f}_r0_fg_nbr_counts.npy')
        num_tcrs = fg_counts.shape[0]
        assert num_tcrs == tcrs.shape[0]

        # read bg counts, compute pvals
        all_bg_counts = {}
        for bg in range(6):
            print('reading bg counts:', bg)
            bg_counts = np.zeros((num_tcrs,))
            for r in range(10):
                bg_counts += np.load(f'{run9prefix}_{fg_tag}_{radius:.1f}_r{r}_bg_'
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
        fgfiles = glob(f'{run10prefix}_{fg_tag}_{radius:.1f}_'
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
            for bg in range(6):
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
        plt.subplot(nrows, ncols, 3*plotno+1)
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

        plt.subplot(nrows, ncols, 3*plotno+2)
        plt.title('color by evalue')
        colors = [-1*np.log10(pvals.iloc[x]['evalue']) for x in list(g)]
        nx.draw_networkx(g, pos, ax=plt.gca(), node_size=10, with_labels=False,
                         # labels=labels,
                         node_color=colors)#, vmin=0, vmax = 48)

        plt.subplot(nrows, ncols, 3*plotno+3)
        plt.title('color by mindist to expanding clone')
        min_xclonedists = xclone_dists.min(axis=1)
        assert min_xclonedists.shape == (pvals.shape[0],)
        colors = [min_xclonedists[x] for x in list(g)]
        nx.draw_networkx(g, pos, ax=plt.gca(), node_size=10, with_labels=False,
                         # labels=labels,
                         node_color=colors, vmin=0, vmax = 48)



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
    bgnum = 4
    max_evalue = 0.1
    num_lines = 50
    target_bg_nbrs = None

    nrows = max(1, int(0.75*np.sqrt(len(fg_files))))
    ncols = (len(fg_files)-1)//nrows + 1
    print(nrows, ncols, fg_files)
    plt.figure(figsize=(ncols*3, nrows*3))
    pngfile = ('/home/pbradley/csdat/raptcr/'
               f'run4run5_pvals_F{len(fg_files)}_{radius:.1f}_bg{bgnum}_'
               f'tbn{target_bg_nbrs}.png')

    run4prefix = '/home/pbradley/csdat/raptcr/slurm/run4/run4'
    run5prefix = '/home/pbradley/csdat/raptcr/slurm/run5/run5'
    run8prefix = '/home/pbradley/csdat/raptcr/slurm/run8/run8'

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
        for bg in range(6):
            print('reading bg counts:', bg)
            bg_counts = np.zeros((num_tcrs,))
            for r in range(10):
                prefix = run8prefix if bg==5 else run4prefix
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
        if 2:
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
            for bg in range(6):
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



if 0: # setup for big cluster calc, brit-vs-brit nbrs
    # this block generates a file containing commands
    # those commands get distributed over the cluster
    #
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    radii = [3.5, 6.5, 12.5, 18.5, 24.5]

    britdir = DATADIR+'phil/britanova/'
    yfvdir = '/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
    df = pd.read_table(britdir+'metadata.txt')


    fg_fnames = sorted(glob(yfvdir+'*count_2.txt.gz')) ; assert len(fg_fnames)==7
    #fg_fnames = [f'{britdir}{x}.gz' for x in df.file_name]
    bg_fnames = [f'{britdir}{x}.gz' for x,y in zip(df.file_name, df.age) if y==0]

    # fnames = glob('/home/pbradley/gitrepos/immune_response_detection/'
    #               'data/phil/britanova/A*gz')
    print(len(fg_fnames), len(bg_fnames))

    runtag = 'run10' ; xargs = ' --max_tcrs 500000 ' # now yfv day15
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
                  '*min_count_2.txt.gz')
    #fnames = glob('/home/pbradley/gitrepos/immune_response_detection/'
    #              'data/phil/britanova/A*gz')
    print(len(fnames))

    runtag = 'run12' ; xargs = ' --num_nbrs 25 '
    #runtag = 'run11' ; xargs = ' --num_nbrs 10 '

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    for fname in fnames:
        for bgnum in range(6):
            ftag = fname.split('/')[-1][:-3]
            for repeat in range(num_repeats):
                outfile_prefix = f'{rundir}{runtag}_{ftag}_r{repeat}'
                cmd = (f'{PY} {EXE} {xargs} --mode nndists_vs_bg --bg_nums {bgnum} '
                       f' --filename {fname} --outfile_prefix {outfile_prefix} '
                       f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
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

    radii = [3.5, 6.5, 12.5, 18.5, 24.5]
    num_repeats = 10

    fnames = glob('/home/pbradley/csdat/yfv/pogorelyy_et_al_2018/Yellow_fever/'
                  '*min_count_2.txt.gz')
    #fnames = glob('/home/pbradley/gitrepos/immune_response_detection/'
    #              'data/phil/britanova/A*gz')
    print(len(fnames))

    runtag = 'run9' ; xargs = ' --max_tcrs 500000 ' # yfv day 15
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
                outfile_prefix = f'{rundir}{runtag}_{ftag}_{radius:.1f}_r{repeat}'
                cmd = (f'{PY} {EXE} {xargs} --mode brit_vs_bg '
                       f' --filename {fname} --radius {radius} '
                       f' --outfile_prefix {outfile_prefix} '
                       f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
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




