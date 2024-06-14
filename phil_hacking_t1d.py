######################################################################################88

# import raptcr
import itertools as it
# from raptcr.constants.hashing import BLOSUM_62
# from raptcr.constants.base import AALPHABET
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from timeit import default_timer as timer
# from sklearn.manifold import MDS
from collections import Counter
from sys import exit
# import sys
from os.path import exists
from glob import glob
from os import mkdir, system
# import random
import phil_functions
from phil_functions import (filter_out_bad_genes_and_cdr3s,
                            compute_nbr_count_pvalues
)




def convert_to_imgt_v_gene( vg ):
    assert vg.startswith('TCRBV')
    assert len(vg) == 10 and vg[-3] == '-'
    dashnum = int(vg[8:10])
    new_vg_yesdash = 'TRBV{}-{}*01'.format( int(vg[5:7]), dashnum )
    if dashnum>1:
        new_vg_nodash = None
    else:
        new_vg_nodash = 'TRBV{}*01'.format( int(vg[5:7]) )
    return new_vg_nodash, new_vg_yesdash

def convert_to_imgt_j_gene( jg ):
    assert jg.startswith('TCRBJ')
    assert len(jg) == 10 and jg[-3] == '-'
    return 'TRBJ{}-{}*01'.format( int(jg[5:7]), int(jg[8:10] ) )


def parse_genes(vgene, jgene, expected_gene_names):
    ''' Returns None on failure
    otherwise returns   (vg, jg)
    '''

    if not vgene or not jgene: # empty: unresolved probably
        print('empty genes {} {}'.format(vgene,jgene))
        return None
    if not vgene.startswith('TCRBV'):
        print('funny vgene: ({})'.format(vgene))
        return None
    vg1, vg2 = convert_to_imgt_v_gene( vgene )
    jg = convert_to_imgt_j_gene( jgene )

    if vg2 in expected_gene_names:
        vg = vg2
    elif vg1 in expected_gene_names:
        vg = vg1
    else:
        print('bad vg:', vgene, vg1, vg2)
        return None

    return vg, jg


if 0: # look at t1d associations of magic seq
    meta = pd.read_csv('/fh/fast/bradley_p/t1d/cohort_1_metadata.csv')
    meta.set_index('subject_id', drop=False, inplace=True)
    hitlines = [x.split() for x in open('/fh/fast/bradley_p/t1d/tmp.cohort1_magic','r')]

    counts = Counter()

    for l in hitlines:
        sid = '_'.join(l[0].split('/')[1].split('_')[:2])
        row = meta.loc[sid]
        counts[row.diabetes_status] += 1

    print(counts.most_common())

    exit()


if 0: # look for hla/t1d associations in cohort_1 hits
    from scipy.stats import hypergeom
    if 0:
        tsvfile = '/home/pbradley/csdat/raptcr/slurm/run45_combo_pvals_evt_1.0.tsv'
        print('reading:', tsvfile)
        df = pd.read_table(tsvfile)

        #min_ratio = 5
        #min_ratio = 4
        min_ratio = 3
        outfile = f'{tsvfile[:-4]}_min_ratio_{min_ratio}.tsv'
        mask = df.fg_bg_nbr_ratio >= min_ratio
        df[mask].to_csv(outfile, sep='\t', index=False)
        print('made:', mask.sum(), outfile)
        exit()

    min_overlap = 5

    meta = pd.read_csv('/fh/fast/bradley_p/t1d/cohort_1_metadata.csv')
    loci = ['A', 'B', 'C', 'DPA1', 'DPB1', 'DQA1',
            'DQB1', 'DRB1', 'DRB3', 'DRB4', 'DRB5']
    meta['filetag'] = 'cohort_1_' + meta.filename.str.slice(0,-4) + '_imgt'


    tsvfile = ('/home/pbradley/csdat/raptcr/slurm/'
               'run45_combo_pvals_evt_1.0.tsv') #full set
    # tsvfile = ('/home/pbradley/csdat/raptcr/slurm/'
    #            'run45_combo_pvals_evt_1.0_min_ratio_3.tsv')
    # tsvfile = ('/home/pbradley/csdat/raptcr/slurm/'
    #            'run45_combo_pvals_evt_1.0_min_ratio_4.tsv')
    # tsvfile = ('/home/pbradley/csdat/raptcr/slurm/'
    #            'run45_combo_pvals_evt_1.0_min_ratio_5.tsv')

    resultsfile = tsvfile[:-4]+'_assocs.tsv'


    df = pd.read_table(tsvfile)
    df['cdr3_core'] = df.junction_aa.str.slice(3,-2)
    df['tcr_id'] = df.v_call + '_' + df.cdr3_core
    df.sort_values('evalue', inplace=True)

    tcrs_df = df.drop_duplicates('tcr_id').set_index('tcr_id')
    tcrs = list(df.drop_duplicates('tcr_id').tcr_id)
    print('num tcrs:', len(tcrs))
    tcr2num = {x:i for i,x in enumerate(tcrs)}

    filetags = sorted(df.filetag.unique())
    assert len(filetags) == 1425 # cohort_1
    filetag2num = {x:i for i,x in enumerate(filetags)}

    df['itcr'] = df.tcr_id.map(tcr2num)
    df['ifile'] = df.filetag.map(filetag2num)

    ntcrs = len(tcrs)
    nfiles = len(filetags)

    occs = np.zeros((ntcrs, nfiles), dtype=bool)

    print('fill occs:', ntcrs, nfiles, df.shape)
    for i,j in zip(df.itcr, df.ifile):
        occs[i,j] = True
    print('DONE fill occs:', ntcrs, nfiles, df.shape)


    #In [23]: meta.diabetes_status.unique()
    #Out[23]: array(['CTRL', 'FDR', 'T1D', 'SDR'], dtype=object)
    meta.sort_values('filetag', inplace=True)
    assert list(meta.filetag) == filetags

    if 2:
        dfl = []
        for tag in meta.diabetes_status.unique():
            tag_occs = np.array(meta.diabetes_status==tag)
            ntag_occs = tag_occs.sum()
            for ii, tcr in enumerate(tcrs):
                overlap = np.sum(tag_occs&occs[ii,:])
                ntcr_occs = occs[ii,:].sum()

                expected = (ntag_occs * ntcr_occs)/nfiles
                if overlap > expected and overlap>min_overlap:

                    pval = hypergeom.sf(overlap-1, nfiles, ntag_occs, ntcr_occs)
                    if pval*ntcrs<=10:
                        print(f'{tag} {ntcrs*pval:9.2e} {overlap:3d} {expected:5.1f} '
                              f'{ntag_occs:3d} {ntcr_occs:4d} {tcr}')
                        dfl.append(dict(
                            tag = tag,
                            pvalue = pval,
                            overlap = overlap,
                            n_tag = ntag_occs,
                            n_tcr = ntcr_occs,
                            total = nfiles,
                            tcr = tcr,
                            num_tcrs = len(tcrs),
                        ))

        t1d_results = pd.DataFrame(dfl)
        t1d_results['tag_type'] = 'disease'
        t1d_results['num_tags'] = len(meta.diabetes_status.unique())

    # now restrict to non-missing hlas
    hla_mask = np.array(~meta.A.isna())
    assert hla_mask.sum() == 1425-65

    moccs = occs[:,hla_mask]
    mmeta = meta[hla_mask]

    total_alleles = 0
    dfl = []
    for locus in loci:
        vals = mmeta[locus]
        a0 = np.array([x.split(';')[0] for x in vals])
        a1 = np.array([x.split(';')[1] for x in vals])
        counts = Counter()
        for v in vals:
            a,b = v.split(';')
            #assert a!=b # NOT TRUE!
            counts[a] += 1
            if b!=a:
                counts[b] += 1
        #assert all(x.count(';')==1 for x in vals)
        #alleles = set(it.chain(*(x.split(';') for x in vals)))
        print(locus, counts.most_common())

        for allele, nallele_occs in counts.most_common():
            if nallele_occs<5:
                break
            total_alleles += 1
            allele_occs = (a0==allele)|(a1==allele)
            assert nallele_occs == allele_occs.sum()

            for ii, tcr in enumerate(tcrs):
                overlap = np.sum(allele_occs&moccs[ii,:])
                ntcr_occs = moccs[ii,:].sum()

                expected = (nallele_occs * ntcr_occs)/hla_mask.sum()
                if overlap > expected and overlap>min_overlap:

                    pval = hypergeom.sf(overlap-1, hla_mask.sum(), nallele_occs,
                                        ntcr_occs)
                    if pval*ntcrs<=10:
                        print(f'{locus:6s} {allele:7s} {ntcrs*pval:9.2e} {overlap:3d} '
                              f'{expected:5.1f} {nallele_occs:3d} {ntcr_occs:4d} {tcr}')

                        dfl.append(dict(
                            tag = locus+"_"+allele,
                            pvalue = pval,
                            overlap = overlap,
                            n_tag = nallele_occs,
                            n_tcr = ntcr_occs,
                            total = hla_mask.sum(),
                            tcr = tcr,
                            num_tcrs = len(tcrs),
                        ))

    hla_results = pd.DataFrame(dfl)
    hla_results['tag_type'] = 'hla'
    hla_results['num_tags'] = total_alleles

    results = pd.concat([t1d_results, hla_results])
    results['evalue'] = results.pvalue * results.num_tcrs * results.num_tags

    cols = ['pvalue', 'evalue', 'fg_nbrs', 'bg_nbrs',
            'fg_bg_nbr_ratio', 'expected_nbrs', 'num_fg_tcrs', 'num_bg_tcrs',
            'v_call', 'j_call', 'junction_aa', 'cdr3_core']
    results = results.join(tcrs_df[cols], on='tcr', rsuffix='_tcr')

    results.to_csv(resultsfile, sep='\t', index=False)

    exit()

if 0: # look at cohort_1 hits
    rundir = f'/home/pbradley/csdat/raptcr/slurm/run45/'

    evalue_threshold = 1.0
    outfile = f'{rundir[:-1]}_combo_pvals_evt_{evalue_threshold}.tsv'

    tsvfiles = sorted(glob(rundir+'*pvals.tsv'))
    assert len(tsvfiles) == 1425

    dfl = []

    for fname in tsvfiles:
        filetag = fname.split('/')[-1][:-10]
        df = pd.read_table(fname)
        df['filetag'] = filetag
        dfl.append(df[df.evalue <= evalue_threshold].copy())
        print(len(dfl), dfl[-1].shape[0], filetag)

        # if len(dfl)>4:
        #     break

    results = pd.concat(dfl)
    results.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

    exit()




if 0: # check on the bg nbr jobs
    runtag = 'run46'
    cmdsfile = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/{runtag}_commands.txt'
    #cmdsfile = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/{runtag}_commands.more.txt'

    for line in open(cmdsfile,'r'):
        l = line.split()
        outfile = l[l.index('--outfile')+1]
        if exists(outfile):
            pass
            #print('good')
        else:
            print('bad', outfile)



    exit()

if 0: # look at some fg/bg comparisons
    cohort = 1

    fg_rundir = f'/home/pbradley/csdat/raptcr/slurm/run45/'
    bg_rundir = f'/home/pbradley/csdat/raptcr/slurm/run46/'

    bdir = '/fh/fast/bradley_p/t1d/'
    files = sorted(glob(f'{bdir}cohort_{cohort}/*tsv'))
    assert len(files) == 1425

    all_totals = {}
    for line in open(bdir+'wc_cohort_1_tsvs.txt','r'):
        l = line.split()
        if l[1] == 'total':
            continue
        ftag = l[1].split('/')[1][:-4]
        all_totals[ftag] = int(l[0])-1 # drop header line
    assert len(all_totals) == 1425

    print('median repsize:', np.median(list(all_totals.values())),
          'mean:', np.mean(list(all_totals.values())))

    total_bg_tcrs = sum(all_totals.values())

    for ii,fname in enumerate(files[:10]):
        ftag = fname.split('/')[-1][:-4]

        tcrs = pd.read_table(fname)
        num_tcrs = tcrs.shape[0]

        assert num_tcrs == all_totals[ftag]

        fg_nbr_counts = np.load(f'{fg_rundir}cohort_{cohort}_{ftag}_nbr_totals.npy')
        bg_nbr_counts = np.load(f'{bg_rundir}cohort_{cohort}_{ftag}_nbr_totals.npy')

        assert fg_nbr_counts.shape == bg_nbr_counts.shape == (num_tcrs,)

        assert np.sum(fg_nbr_counts<0.5)==0 # no zeros
        fg_nbr_counts -= 1. # self nbrs

        df = compute_nbr_count_pvalues(
            fg_nbr_counts, bg_nbr_counts, total_bg_tcrs,
            num_fg_tcrs=total_bg_tcrs,
        )

        exit()



    exit()


if 0: # gather the fg nbr-count info
    cohorts = [1]
    runtag = 'run46'
    #runtag = 'run45'
    bdir = '/fh/fast/bradley_p/t1d/'
    batch_size = 20
    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'

    for cohort in cohorts:

        files = sorted(glob(f'{bdir}cohort_{cohort}/*tsv'))

        num_batches = (len(files)-1)//batch_size + 1
        print('num_files:', len(files), 'num_batches:', num_batches)

        for fg_file in files:
            fg_tag = fg_file.split('/')[-1][:-4]
            mydir = f'{rundir}cohort_{cohort}_{fg_tag}/'
            print(files.index(fg_file), mydir)
            nbr_counts = None
            partial = False
            for b in range(num_batches):
                start, stop = b*batch_size, (b+1)*batch_size
                nbrsfile = f'{mydir}cohort_{cohort}_{fg_tag}_{b}_{start}_{stop}.npy'
                if not exists(nbrsfile):
                    partial = True
                    break
                #assert exists(nbrsfile)
                counts = np.load(nbrsfile)
                if b==0:
                    nbr_counts = counts
                else:
                    nbr_counts += counts
            if partial:
                print('partial!', fg_tag)
                break
            outfile = f'{rundir}cohort_{cohort}_{fg_tag}_nbr_totals.npy'
            np.save(outfile, nbr_counts)
            print('makde:', outfile)

    exit()

if 0: # accumulate fg/bg counts, compute pvals
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    runtag = 'run47'
    xargs = ' --mode t1d_step2 --fg_runtag run45 --bg_runtag run46 '
    cohorts = [1]

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    bdir = '/fh/fast/bradley_p/t1d/'
    bgdir = bdir+'background/'

    for cohort in cohorts:

        files = sorted(glob(f'{bdir}cohort_{cohort}/*tsv'))
        if cohort==1:
            assert len(files) == 1425

        for fname in files:
            filetag = f'cohort_{cohort}_' + fname.split('/')[-1][:-4]
            outprefix = f'{rundir}{runtag}_{filetag}'

            cmd = (f'{PY} {EXE} {xargs} --filetag {filetag} '
                   f' > {outprefix}.log 2> {outprefix}.err')
            out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)
    exit()

if 0: # get background nbr counts
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    runtag = 'run46'
    batch_size = 20 # pairs per run, smallish for restart queue
    cohorts = [1]
    xargs = (' --v_column v_call --j_column j_call --cdr3aa_column junction_aa '
             ' --cdr3nt_column junction --radius 12.5 ')

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    bdir = '/fh/fast/bradley_p/t1d/'
    bgdir = bdir+'background/'

    for cohort in cohorts:

        files = sorted(glob(f'{bdir}cohort_{cohort}/*tsv'))
        all_bg_files = [x.replace(bdir, bgdir) for x in files]
        assert all(exists(x) for x in all_bg_files)

        num_batches = (len(files)-1)//batch_size + 1
        print('num_files:', len(files), 'num_batches:', num_batches)

        for fg_file in files:
            fg_tag = fg_file.split('/')[-1][:-4]
            mydir = f'{rundir}cohort_{cohort}_{fg_tag}/'
            if not exists(mydir):
                mkdir(mydir)
            print(files.index(fg_file), mydir)
            for b in range(num_batches):
                start, stop = b*batch_size, (b+1)*batch_size
                bg_files = all_bg_files[start:stop]
                outfile = f'{mydir}cohort_{cohort}_{fg_tag}_{b}_{start}_{stop}.npy'
                cmd = (f'{PY} {EXE} {xargs} --mode nbr_counts '
                       f' --fg_filename {fg_file} --outfile {outfile} '
                       f' --bg_filenames {" ".join(bg_files)} '
                       f' > {outfile[:-4]}.log 2> {outfile[:-4]}.err')
                out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)
    exit()


if 0: # get foreground nbr counts
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    runtag = 'run45'
    batch_size = 20 # pairs per run, smallish for restart queue
    cohorts = [1]
    xargs = (' --v_column v_call --j_column j_call --cdr3aa_column junction_aa '
             ' --cdr3nt_column junction --radius 12.5 ')

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    bdir = '/fh/fast/bradley_p/t1d/'

    for cohort in cohorts:

        files = sorted(glob(f'{bdir}cohort_{cohort}/*tsv'))

        num_batches = (len(files)-1)//batch_size + 1
        print('num_files:', len(files), 'num_batches:', num_batches)

        for fg_file in files:
            fg_tag = fg_file.split('/')[-1][:-4]
            mydir = f'{rundir}cohort_{cohort}_{fg_tag}/'
            if not exists(mydir):
                mkdir(mydir)
            print(files.index(fg_file), mydir)
            for b in range(num_batches):
                start, stop = b*batch_size, (b+1)*batch_size
                bg_files = files[start:stop]
                outfile = f'{mydir}cohort_{cohort}_{fg_tag}_{b}_{start}_{stop}.npy'
                cmd = (f'{PY} {EXE} {xargs} --mode nbr_counts '
                       f' --fg_filename {fg_file} --outfile {outfile} '
                       f' --bg_filenames {" ".join(bg_files)} '
                       f' > {outfile[:-4]}.log 2> {outfile[:-4]}.err')
                out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)
    exit()



if 0: # setup for running on cluster: make background files
    PY = '/home/pbradley/miniconda3/envs/raptcr/bin/python'
    EXE = '/home/pbradley/gitrepos/immune_response_detection/phil_running.py'

    runtag = 'run44'
    xargs = (' --v_column v_call --j_column j_call --cdr3aa_column junction_aa '
             ' --cdr3nt_column junction ')

    rundir = f'/home/pbradley/csdat/raptcr/slurm/{runtag}/'
    if not exists(rundir):
        mkdir(rundir)

    cmds_file = f'{rundir}{runtag}_commands.txt'
    assert not exists(cmds_file)
    out = open(cmds_file,'w')

    bdir = '/fh/fast/bradley_p/t1d/'
    outdir = bdir+'background/'
    assert exists(outdir)

    files = glob(bdir+'cohort_1/*tsv')+glob(bdir+'cohort_2/*tsv')
    print('num files:', len(files))

    for fname in files:
        outfile = fname.replace(bdir, outdir)

        cmd = (f'{PY} {EXE} {xargs} --mode make_bg '
               f' --filename {fname} --outfile {outfile} '
               f' > {outfile[:-4]}.log 2> {outfile[:-4]}.err')
        out.write(cmd+'\n')
    out.close()
    print('made:', cmds_file)
    exit()




if 0: # parse the adaptive files, assign correct gene names, figure out
    # nucleotide seqs
    import tcrdist_old
    from tcrdist_old.all_genes import all_genes # for recognized genes
    from tcrdist_old.translation import get_translation
    from tcrdist_old.tcr_sampler import get_j_cdr3_nucseq

    organism = 'human'

    expected_gene_names = set(all_genes[organism].keys())

    expected_bad_genes = set([
        'TCRBV07-05', 'TCRBV22-01', 'TCRBV08-02', 'TCRBV08-01', 'TCRBV05-02',
        'TCRBJ02-02P',
    ]) # all alleles are pseudogenes

    vsubs = [
        ('TCRBV03-01/03-02', 'TCRBV03-01'),
        ('TCRBV06-02/06-03', 'TCRBV06-02'),
        ('TCRBV12-03/12-04', 'TCRBV12-03'),
    ]

    olddir = '/loc/no-backup/pbradley/share/t1d/'
    newdir = '/home/pbradley/csdat/raptcr/t1d/'

    #files = glob(olddir+'cohort_2/*tsv')
    #outdir = newdir+'cohort_2/'
    files = glob(olddir+'cohort_1/*tsv')
    outdir = newdir+'cohort_1/'
    if not exists(outdir):
        mkdir(outdir)

    print(len(files))

    for fname in files:
        df = pd.read_table(fname)

        for a,b in vsubs:
            df['v_gene'] = df.v_gene.replace(a,b)

        namask = (df.v_gene.isna() | df.j_gene.isna() | df.amino_acid.isna() |
                  df.templates.isna() | df.rearrangement.isna())
        badmask = (df.v_gene.isin(expected_bad_genes)|
                   df.j_gene.isin(expected_bad_genes))
        unresmask = (df.v_gene == 'unresolved') | (df.j_gene == 'unresolved')
        bvamask = (df.v_gene == 'TCRBVA-01')
        ormask = df.v_gene.str.contains('-or')

        print('badmask:', badmask.sum(), 'unresmask:', unresmask.sum(),
              'bvamask:', bvamask.sum(), 'namask:', namask.sum(),
              'ormask:', ormask.sum(), fname)

        df = df[~(namask|badmask|unresmask|bvamask|ormask)]

        dfl = []
        for counter, l in enumerate(df.itertuples()):
            if counter%25000==0:
                print(counter, files.index(fname), df.shape[0], fname)
            if l.frame_type != 'In':
                print('error skip bad frame:', l.frame_type, fname)
                continue
            readseq = l.rearrangement
            cdr3aa = l.amino_acid

            pos = None
            for offset in range(3):
                protseq = get_translation(readseq, '+{}'.format(offset+1))
                if cdr3aa in protseq:
                    pos = 3*protseq.index(cdr3aa) + offset
                    break
            else:
                print('error cdr3aa not in readseq', cdr3aa, readseq, fname)
                continue

            cdr3nt = readseq[pos:pos+3*len(cdr3aa)]
            assert get_translation(cdr3nt, '+1') == cdr3aa

            try:
                res = parse_genes(l.v_gene, l.j_gene, expected_gene_names)
                if res is None:
                    print('error parse fail:', l.v_gene, l.j_gene, fname)
                    continue
            except:
                print('error unknown parse_genes error:', l, fname)
                continue

            dfl.append(dict(
                v= res[0],
                j= res[1],
                cdr3aa = cdr3aa,
                cdr3nt = cdr3nt,
                count = l.templates,
            ))

        newdf = pd.DataFrame(dfl)

        v_column, j_column, cdr3_column, organism, chain = 'v','j','cdr3aa','human','B'
        newdf = filter_out_bad_genes_and_cdr3s(
            newdf, v_column, cdr3_column, organism, chain, j_column=j_column)

        newdf.rename(columns={
            'v':'v_call',
            'j':'j_call',
            'cdr3aa':'junction_aa',
            'cdr3nt':'junction',
            'count':'clone_count',
            }, inplace=True)

        fb = fname.split('/')[-1]
        outfile = f'{outdir}{fb[:-4]}_imgt.tsv'
        newdf.to_csv(outfile, sep='\t', index=False)
        print('made:', df.shape[0]-newdf.shape[0], outfile)
