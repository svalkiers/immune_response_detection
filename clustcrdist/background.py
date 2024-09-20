import numpy as np
import pandas as pd
import itertools
import pkg_resources

from concurrent.futures import ThreadPoolExecutor   
from os.path import dirname, abspath, join
from collections import Counter

from .constants.modules.tcrdist.tcr_sampler import parse_tcr_junctions,  resample_shuffled_tcr_chains
from .constants.preprocessing import format_chain
from .constants.base import AALPHABET
from .pgen import generate_sequences


# ROOT = dirname(dirname(dirname(abspath(__file__))))
# MODELS = join(ROOT, 'clustcrdist/constants/modules/olga/default_models/human_T_beta/')
# DATADIR = join(ROOT, 'clustcrdist/constants/data/')
MODELS = pkg_resources.resource_filename('clustcrdist', 'constants/modules/olga/default_models/human_T_beta/')
DATADIR = pkg_resources.resource_filename('clustcrdist', 'constants/data/')

def get_vfam(df, vcol='v_call'):
    '''
    Extract V gene family information from V gene column.
    '''
    return df[vcol].apply(lambda x: x.split('*')[0].split('-')[0])

def get_jfam(df, jcol='j_call'):
    '''
    Extract J gene family information from J gene column.
    '''
    return df[jcol].apply(lambda x: x.split('*')[0].split('-')[0])

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
    all_genes_df = pd.read_table(DATADIR+'combo_xcr.tsv')
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

class BackgroundModel():
    def __init__(
        self,
        repertoire,
        factor:int=10,
        # chain:str="B",
        organism:str='human',
        v_column:str='v_call',
        j_column:str='j_call',
        cdr3nt_column:str='junction',
        cdr3aa_column:str='junction_aa', 
        verbose=False
        ):
        '''
        Class for creating synthetic backgrounds that match properties of the
        input repertoire, including V/J gene distribution, and CDR3 length.
        '''
        self.repertoire = repertoire
        self.factor = factor
        # self.chain = format_chain(chain)
        self.organism = organism
        self.junctions = None
        self.size = len(self.repertoire)
        self.n = self.size * self.factor
        self._setup_gene_ref()

        # Prepare gene columns
        self.v_column = v_column
        self.j_column = j_column
        self.cdr3nt_column = cdr3nt_column
        self.cdr3aa_column = cdr3aa_column
        # self._add_v_gene_column()
        # self._add_j_gene_column()
        # self.v_genes = self.repertoire.v_gene.unique()

        self.verbose = verbose

    def _add_v_gene_column(self):
        '''
        Add a V gene column to the repertoire file by parsing the V allele (v_call) column.
        '''
        self.repertoire['v_gene'] = self.repertoire.v_call.apply(lambda x: x.split('*')[0])

    def _add_j_gene_column(self):
        '''
        Add a J gene column to the repertoire file by parsing the J allele (j_call) column.
        '''
        self.repertoire['j_gene'] = self.repertoire.j_call.apply(lambda x: x.split('*')[0])

    def _setup_gene_ref(self):
        '''
        Setup reference database 
        !NOTE! the following V alleles have been removed from the reference:
            - TRBV7-2*03 (71)
            - TRBV7-3*04 (76)
            - TRBV15*03 (22)
            - TRBV4-3*02 (48)
        '''
        func_v_genes = pd.read_csv(join(MODELS,'functional_V_genes.csv'), index_col=[0])
        func_j_genes = pd.read_csv(join(MODELS,'functional_J_genes.csv'), index_col=[0])
        self.v_gene_ref = {v:func_v_genes[func_v_genes.gene==v].allele.to_list() for v in func_v_genes.gene.unique()}
        self.j_gene_ref = {j:func_j_genes[func_j_genes.gene==j].allele.to_list() for j in func_j_genes.gene.unique()}
        self.v_allele_ref = dict(zip(func_v_genes.allele, func_v_genes.index))
        self.j_allele_ref = dict(zip(func_j_genes.allele, func_j_genes.index))


    def _parse_junctions(self, chain):
        '''
        setup for the "resample_background_tcrs" function by parsing
        the V(D)J junctions in the foreground tcr set

        returns a dataframe with info
        '''

        chain = format_chain(chain)

        # tcrdist parsing function expects paired tcrs as list of tuples of tuples
        tcr_columns = [
            self.v_column, 
            self.j_column, 
            self.cdr3aa_column, 
            self.cdr3nt_column
            ]
        if 'junction' in self.repertoire.columns:
            self.repertoire[self.cdr3nt_column] = self.repertoire[self.cdr3nt_column].str.lower()
        else:
            self.repertoire['cdr3a_nucseq'] = self.repertoire.cdr3a_nucseq.str.lower()
            self.repertoire['cdr3b_nucseq'] = self.repertoire.cdr3b_nucseq.str.lower()
        if chain in ['A','G']:
            print(chain)
            if self.repertoire.v_call.apply(lambda x: x[:3]).nunique() > 1:
                ag = self.repertoire[self.repertoire['locus'] == f'TR{chain}']
                ag = ag[tcr_columns]
                tcr_tuples = ag.itertuples(name=None, index=None)
            else:
                print('Single chain detected.')
                tcr_tuples = self.repertoire[tcr_columns].itertuples(name=None, index=None)
            tcr_tuples = zip(tcr_tuples, itertools.repeat(None))
        elif chain in ['B','D']:
            print(chain)
            if self.repertoire.v_call.apply(lambda x: x[:3]).nunique() > 1:
                bd = self.repertoire[self.repertoire['locus'] == f'TR{chain}']
                bd = bd[tcr_columns]
                tcr_tuples = bd.itertuples(name=None, index=None)
            else:
                print('Single chain detected.')
                tcr_tuples = self.repertoire[tcr_columns].itertuples(name=None, index=None)
            tcr_tuples = zip(itertools.repeat(None), tcr_tuples)
        elif chain in ['AB','GD']:
            print(chain)
            # Check formatting
            airr_cols = ['v_call','j_call','junction_aa','junction']
            if all([i in self.repertoire.columns for i in airr_cols]):
                print('Detected AIRR format')
                ag = self.repertoire[self.repertoire['locus'] == f'TR{list(chain)[0]}']
                ag = ag[tcr_columns]
                bd = self.repertoire[self.repertoire['locus'] == f'TR{list(chain)[1]}']
                bd = bd[tcr_columns]
            else:
                print('No AIRR format detected, proceeding with paired TCRdist format.')
                ag = self.repertoire[['va','ja','cdr3a','cdr3a_nucseq']]
                bd = self.repertoire[['vb','jb','cdr3b','cdr3b_nucseq']]
            tcr_tuples_ag = ag.itertuples(name=None, index=None)
            tcr_tuples_bd = bd.itertuples(name=None, index=None)
            tcr_tuples = zip(tcr_tuples_ag, tcr_tuples_bd)

        self.junctions = parse_tcr_junctions(self.organism, list(tcr_tuples))
        #junctions = add_vdj_splits_info_to_junctions(junctions)
        return self.junctions

    def resample_background_tcrs(
        self,
        chain,
        preserve_vj_pairings=False, # consider setting True for alpha chain
        return_nucseq_srcs=False, # for debugging
        verbose=False
        ):
        '''
        Resample shuffled background sequences, number will be equal to size of
        foreground repertore, ie junctions.shape[0]

        junctions is a dataframe with information about the V(D)J junctions in the
        foreground tcrs. Created by the function "_parse_junctions()".

        returns a list of tuples [(v,j,cdr3aa,cdr3nt), ...] of length = junctions.shape[0]
        '''

        chain = format_chain(chain)

        if self.junctions is None:
            self._parse_junctions()

        multiplier = 3 # so we have enough to match distributions
        bg_tcr_tuples, src_junction_indices = resample_shuffled_tcr_chains(
            self.organism, multiplier * self.junctions.shape[0], chain, self.junctions,
            preserve_vj_pairings=preserve_vj_pairings,
            return_src_junction_indices=True
            )

        fg_nucseq_srcs = list(self.junctions[f'cdr3{chain.lower()}_nucseq_src'])

        resamples = []
        for tcr, inds in zip(bg_tcr_tuples, src_junction_indices):
            if len(resamples)%500000==0:
                if self.verbose:
                    print(f'{BackgroundModel.resample_background_tcrs.__name__}: build nucseq_srclist', len(resamples),
                        len(bg_tcr_tuples))
            v,j,cdr3aa,cdr3nt = tcr
            v_nucseq = fg_nucseq_srcs[inds[0]]
            j_nucseq = fg_nucseq_srcs[inds[1]]
            nucseq_src = v_nucseq[:inds[2]] + j_nucseq[inds[2]:]
            assert len(tcr[3]) == len(nucseq_src)   
            resamples.append((v, j, cdr3aa, cdr3nt, nucseq_src))


        # try to match lengths first
        fg_lencounts = Counter(len(x) for x in self.junctions['cdr3'+chain.lower()])

        N = self.junctions.shape[0]
        good_resamples = resamples[:N]
        bad_resamples = resamples[N:]

        bg_lencounts = Counter(len(x[2]) for x in good_resamples)
        all_lencounts = Counter(len(x[2]) for x in resamples)
        if not all(all_lencounts[x]>=fg_lencounts[x] for x in fg_lencounts):
            if self.verbose:
                print('dont have enough of all lens')

        tries = 0
        too_many_tries = 10*self.junctions.shape[0]

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
        if self.verbose:
            print('desirable_Ncounts:', desirable_Ncounts)
        bad_resamples = [x for x in bad_resamples if x[4].count('N') in desirable_Ncounts]

        # now try to match the N insertion distributions, while preserving the
        # length distributions
        bad_ncounts = Counter(x[4].count('N') for x in bad_resamples)
        all_ncounts = Counter(x[4].count('N') for x in resamples)

        tries = 0
        too_many_tries = 10*self.junctions.shape[0]
        too_many_inner_tries = 1000

        for num in desirable_Ncounts:
            if self.verbose:
                print('Ns:', num, 'fg_ncounts:', fg_ncounts[num],
                    'bg_ncounts:', bg_ncounts[num], 'bad_ncounts:', bad_ncounts[num],
                    'sum:', bg_ncounts[num] + bad_ncounts[num])
            if fg_ncounts[num] > all_ncounts[num]:
                if self.verbose:
                    print(f'{BackgroundModel.resample_background_tcrs.__name__} dont have enough Ns:',
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
                    if self.verbose:
                        print('ran out of desirable_Ncounts:', desirable_Ncounts)
                    break

        if self.verbose:
            print('final Ndevs:', end=' ')
        for ii in range(100):
            if fg_ncounts[ii] != bg_ncounts[ii]:
                if self.verbose:
                    print(ii, fg_ncounts[ii]-bg_ncounts[ii], end=' ')
        # print()
        if self.verbose:
            print('final Ldevs:', end=' ')
        for ii in range(100):
            if fg_lencounts[ii] != bg_lencounts[ii]:
                if self.verbose:
                    print(ii, fg_lencounts[ii]-bg_lencounts[ii], end=' ')
        # print()

        good_tcrs = [x[:4] for x in good_resamples]
        if return_nucseq_srcs:
            good_nucseq_srcs = [x[4] for x in good_resamples]
            return good_tcrs, good_nucseq_srcs
        else:
            return good_tcrs

    def process_background(self, chain) -> pd.DataFrame:
        return pd.DataFrame(self.resample_background_tcrs(chain=chain))
    

    def shuffle(self, chain):
        """
        Shuffles T-cell receptor (TCR) sequences for a specified chain or pair of chains.

        This function generates a background dataset of TCR sequences by resampling and shuffling existing sequences.
        It supports shuffling for individual chains ('A', 'B', 'G', 'D') or a pair of chains ('AB', 'GD').
        For individual chains, it parses junctions and processes the background for the specified chain.
        For the 'AB' pair, it resamples background TCRs for both chains while preserving V-J pairings (alpha),
        and filters out sequences with bad genes or CDR3 regions.

        Parameters:
        - chain (str): The TCR locus to shuffle. Accepted values are 'A', 'B', 'G' or 'D' for single chains,
         and 'AB' or 'GD' for paired chains.

        Returns:
        - pd.DataFrame: A DataFrame containing the shuffled TCR sequences. The columns of the DataFrame
        depend on the input chain(s). For individual chains, the columns are V gene, J gene, CDR3 amino acid sequence,
        and CDR3 nucleotide sequence. For 'AB' pairs, the columns include these for both chains A and B.
        """

        # Standardize chain selection format
        chain = format_chain(chain)

        if chain in ['A','B','G','D']:
            self._parse_junctions(chain=chain)
            background = [self.process_background(chain=chain) for i in range(self.factor)]
            background = pd.concat(background)
            background.columns = [self.v_column, self.j_column, self.cdr3aa_column, self.cdr3nt_column]
            return background

        chain = format_chain(chain)
        if chain == 'AB':
            self._parse_junctions(chain=chain)
            dfl = []
            for r in range(self.factor):
                # Get alpha and beta background
                tups_a = self.resample_background_tcrs(chain='A',preserve_vj_pairings=True)
                tups_b = self.resample_background_tcrs(chain='B')
                bg_tcrs = pd.DataFrame([
                    dict(va=va, ja=ja, cdr3a=cdr3a, cdr3a_nucseq=cdr3a_nucseq,
                        vb=vb, jb=jb, cdr3b=cdr3b, cdr3b_nucseq=cdr3b_nucseq)
                        for ((va,ja,cdr3a,cdr3a_nucseq), (vb,jb,cdr3b,cdr3b_nucseq)) in zip(tups_a, tups_b)
                        ])
                # Remove any TCRs with bad (ORF, pseudo) genes or non-AA characters in CDR3
                bg_tcrs = filter_out_bad_genes_and_cdr3s(
                    bg_tcrs, 'va', 'cdr3a', 'human', 'A', j_column='ja')
                bg_tcrs = filter_out_bad_genes_and_cdr3s(
                    bg_tcrs, 'vb', 'cdr3b', 'human', 'B', j_column='jb')
                dfl.append(bg_tcrs)
            return pd.concat(dfl).reset_index(drop=True)

    def olga(self, chain):
        '''
        Generates a background repertoire using the OLGA model.
        Currently only supports alpha-beta TCRs since there is
        no OLGA model for gamma-delta TCRs.
        '''
        # Standardize chain selection format
        chain = format_chain(chain)
        # Single chain
        if chain in ['A','B']:
            nsample = self.repertoire[self.repertoire['locus']==f'TR{chain}'].shape[0] * self.factor
            return generate_sequences(nsample, chain=chain)
        # Paired chains
        elif chain == 'AB':
            nsample = self.repertoire[self.repertoire['locus']==f'TRA'].shape[0] * self.factor
            alpha = generate_sequences(nsample, chain='A')
            alpha = alpha[['junction','junction_aa','v_call','j_call']]
            alpha.columns = ['cdr3a_nucseq','cdr3a','va','ja']
            beta = generate_sequences(nsample, chain='B')
            beta = beta[['junction','junction_aa','v_call','j_call']]
            beta.columns = ['cdr3b_nucseq','cdr3b','vb','jb']
            paired = pd.concat([alpha,beta],axis=1)
            paired = paired[['va', 'ja', 'cdr3a', 'cdr3a_nucseq', 'vb', 'jb', 'cdr3b','cdr3b_nucseq']]
            return paired


    # def test_background(self):

    #     if self.chain in ['A','B','G','D']:
    #         return [self.process_background() for i in range(self.factor)]
    #     else:
    #         if chain == 'AB':
    #             a = 'A'
    #             b = 'B'
    #         print(a,b)
    #         dfl = []
    #         for r in range(self.factor):
    #             tups_a = self.resample_background_tcrs(chain='A')
    #             tups_b = self.resample_background_tcrs(chain='B')
    #             bg_tcrs = pd.DataFrame([
    #                 dict(va=va, ja=ja, cdr3a=cdr3a, cdr3a_nucseq=cdr3a_nucseq,
    #                     vb=vb, jb=jb, cdr3b=cdr3b, cdr3b_nucseq=cdr3b_nucseq)
    #                     for ((va,ja,cdr3a,cdr3a_nucseq), (vb,jb,cdr3b,cdr3b_nucseq)) in zip(tups_a, tups_b)
    #                     ])
    #             bg_tcrs = filter_out_bad_genes_and_cdr3s(
    #                 bg_tcrs, 'va', 'cdr3a', 'human', 'A', j_column='ja')
    #             bg_tcrs = filter_out_bad_genes_and_cdr3s(
    #                 bg_tcrs, 'vb', 'cdr3b', 'human', 'B', j_column='jb')
    #             dfl.append(bg_tcrs)
    #         return pd.concat(dfl).reset_index(drop=True)

    # def shuffled_background(self, num_workers=1, destination=None):
    #     '''
    #     Creates a background repertoire by reshuffling the input repertoire
    #     n times (n = factor).

    #     num_workers: number of CPUs allocated
    #     '''
    #     # If num_workers > 1, use multiprocessing
    #     if num_workers > 1:
    #         import multiprocessing as mp 
    #         with mp.Pool(processes=num_workers) as pool:
    #             backgrounds = pool.starmap(self.process_background, [() for _ in range(self.factor)])
    #     else:
    #         backgrounds = [self.process_background() for i in range(self.factor)]
    #     # Combine reshuffled repertoires
    #     bg = pd.concat(backgrounds)
    #     bg.columns = [self.v_column, self.j_column, self.cdr3aa_column, self.cdr3nt_column]
    #     # Option: save to disk
    #     if destination is not None:
    #         bg.to_csv(destination, sep="\t", index=False)
    #     return bg

    # def batch_shuffling(self, destination):
    #     '''
    #     Performs background shuffling in batches and iteratively stores
    #     output on disk. 
    #     '''
    #     for i in range(self.factor):
    #         background = self.process_background()
    #         if i == 0:
    #             background.columns = [self.v_column, self.j_column, self.cdr3aa_column, self.cdr3nt_column]
    #             background.to_csv(destination, sep="\t", index=False)
    #         else:
    #             background.to_csv(destination, sep="\t", mode="a", index=False, header=False)