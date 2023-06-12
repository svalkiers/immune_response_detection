import pandas as pd
import numpy as np
import math
import multiprocessing as mp
import parmap
import random
import itertools

from collections import Counter
from .constants.modules.olga import load_model, sequence_generation
from .constants.modules.olga.sequence_generation import SequenceGenerationVDJ
from .constants.modules.olga import sequence_generation, load_model, olga_directed
from .constants.modules.tcrdist.tcr_sampler import parse_tcr_junctions

from os.path import dirname, abspath, join

ROOT = dirname(dirname(dirname(abspath(__file__))))
MODELS = join(ROOT,'immune_response_detection/raptcr/constants/modules/olga/default_models/human_T_beta/')

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

def match_vj_distribution(n:int, foreground:pd.DataFrame, background:pd.DataFrame=None):
    '''
    Takes a random sample from a background dataset, while matching the V and J gene
    distribution in the foreground dataset.

    Parameters
    ----------
    n : int
        Sample size.
    foreground : pd.DataFrame
        Foreground dataset.
    background : pd.DataFrame
        Background dataset. Use default when none specified.
    '''
    # Use default background when none specified
    if background is None:
        if n > 1e6:
            background = sample_tcrs(int(n))
        else:
            background = sample_tcrs(int(1e6))
        # background = pd.read_csv('./raptcr/datasets/1m_sequences.tsv', sep='\t')
    else:
        cols = background.columns
        assert 'v_call' in cols and 'j_call' in cols and 'junction_aa' in cols,\
            'background must contain at least the following columns: v_call, j_call, junction_aa'
    
    # Extract V and J family frequencies
    background['vfam'] = get_vfam(background)
    foreground['vfam'] = get_vfam(foreground)
    background['jfam'] = get_jfam(background)
    foreground['jfam'] = get_jfam(foreground)
    vfreqs = foreground.vfam.value_counts()/foreground.vfam.value_counts().sum()
    jfreqs = foreground.jfam.value_counts()/foreground.jfam.value_counts().sum()
    vfam_counts = dict(np.round(vfreqs*n, 0).astype(int))
    jfam_counts = dict(np.round(jfreqs*n, 0).astype(int))
    actual_n = min(sum(list(vfam_counts.values())), sum(list(jfam_counts.values())))

    vgenes = pd.concat([background[background.vfam==v].v_call.sample(vfam_counts[v], replace=True) for v in vfam_counts])
    jgenes = pd.concat([background[background.jfam==j][['j_call','junction_aa']].sample(jfam_counts[j], replace=True) for j in jfam_counts])
    
    if actual_n < n:
        vgenes = vgenes.sample(actual_n)
        jgenes = jgenes.sample(actual_n)
    else:
        pass

    return pd.concat([vgenes.reset_index(drop=True), jgenes.reset_index(drop=True)], axis=1).dropna()

def _setup_olga_models():

    params_file_name = './raptcr/constants/modules/olga/default_models/human_T_beta/model_params.txt'
    marginals_file_name = './raptcr/constants/modules/olga/default_models/human_T_beta/model_marginals.txt'
    V_anchor_pos_file ='./raptcr/constants/modules/olga/default_models/human_T_beta/V_gene_CDR3_anchors.csv'
    J_anchor_pos_file = './raptcr/constants/modules/olga/default_models/human_T_beta/J_gene_CDR3_anchors.csv'

    genomic_data = load_model.GenomicDataVDJ()
    genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

    generative_model = load_model.GenerativeModelVDJ()
    generative_model.load_and_process_igor_model(marginals_file_name)
    
    return genomic_data, generative_model

def generate_olga_sequences(n, genomic_data, generative_model):
    seq_gen_model = SequenceGenerationVDJ(generative_model, genomic_data)
    colnames = ['junction','junction_aa','v_call','j_call']
    df = pd.DataFrame([seq_gen_model.gen_rnd_prod_CDR3() for i in range(n)], columns=colnames)
    df['v_call'] = df['v_call'].apply(lambda x: genomic_data.genV[x][0])
    df['j_call'] = df['j_call'].apply(lambda x: genomic_data.genJ[x][0])
    df['v_gene'] = df['v_call'].apply(lambda x: x.split('*')[0])
    df['j_gene'] = df['j_call'].apply(lambda x: x.split('*')[0])
    return df

def generate_olga_sequences_multi(total, chunks, genomic_data=None, generative_model=None, ncpus=1):
    if genomic_data is None and generative_model is None:
        genomic_data, generative_model = _setup_olga_models()
    with mp.Pool(ncpus) as pool:
        seq = parmap.map(
            generate_olga_sequences,
            [chunks] * int(total/chunks),
            genomic_data,
            generative_model,
            pm_parallel=True,
            pm_pool=pool
            )
    return pd.concat(seq).reset_index(drop=True)

def get_frequency_distributions(df):
    vfreq = dict(df.v_call.apply(lambda x: x.split('*')[0]).value_counts()/len(df))
    jfreq = dict(df.j_call.apply(lambda x: x.split('*')[0]).value_counts()/len(df))
    cdr3len = dict(df.junction_aa.str.len().value_counts()/len(df))
    return vfreq, jfreq, cdr3len

def get_prob(feature, probabilities):
    try:
        p = probabilities[feature]
    except KeyError:
        p = 0
    return p

def _update_probabilities(n_left, n_desired):
    return {i:n_desired[i]/n_left if n_desired[i]/n_left > 0 else 0 for i in n_desired}

def _update_n_desired(counts, n_desired):
    for feature in counts.index:
        n_desired[feature] = n_desired[feature]-counts[feature] 
    return n_desired

def matched_property_sampling(repertoire, total, chunksize, ncpus=8):
    
    background = []
    genomic_data, generative_model = _setup_olga_models()

    # Initialization
    v_freq, j_freq, len_freq = get_frequency_distributions(repertoire)
    v_desired = {i:math.ceil(v_freq[i]*total) for i in v_freq}
    j_desired = {i:math.ceil(j_freq[i]*total) for i in j_freq}
    len_desired = {i:math.ceil(len_freq[i]*total) for i in len_freq}

    start = 0
    stop = total + 1
    step = chunksize
    for i in range(start, stop, step):
        if i % 100000 == 0:
            print(i)
        # Update probabilites
        p_v = _update_probabilities(n_left=total, n_desired=v_desired)
        p_j = _update_probabilities(n_left=total, n_desired=j_desired)
        p_len = _update_probabilities(n_left=total, n_desired=len_desired)
        # Generate new batch of background sequences and assign weights
        sample = generate_olga_sequences_multi(
            total=chunksize*5, chunks=int(chunksize/5), 
            genomic_data=genomic_data, generative_model=generative_model,
            ncpus=ncpus
            )
        sample['p_l'] = sample.junction_aa.apply(lambda cdr3: get_prob(len(cdr3),p_len))
        sample['p_v'] = sample.v_gene.apply(lambda v: get_prob(v,p_v))
        sample['p_j'] = sample.j_gene.apply(lambda j: get_prob(j,p_j))
        sample['w_tcr'] = sample['p_l'] * sample['p_v'] * sample['p_j']
        weights = sample.w_tcr.to_list()
        # while sum(weights) <= 0:
        #     sample = generate_olga_sequences_multi(
        #         total=chunksize*10, chunks=int(chunksize),
        #         genomic_data=genomic_data, generative_model=generative_model,
        #         ncpus=ncpus
        #         )
        #     sample['p_l'] = sample.junction_aa.apply(lambda cdr3: get_prob(len(cdr3),p_len))
        #     sample['p_v'] = sample.v_gene.apply(lambda v: get_prob(v,p_v))
        #     sample['p_j'] = sample.j_gene.apply(lambda j: get_prob(j,p_j))
        #     sample['w_tcr'] = sample['p_l'] * sample['p_v'] * sample['p_j']
        #     weights = sample.w_tcr.to_list()
        if sum(weights) <= 0:
            return p_v, p_j, p_len
        tcr_ids = sample.index
        # Sample from background
        selected = random.choices(tcr_ids, weights=weights, k=chunksize)
        sampled_tcrs = sample.loc[selected]
        background.append(sampled_tcrs)
        # Update records
        v_desired = _update_n_desired(sampled_tcrs.v_gene.value_counts(), v_desired)
        j_desired = _update_n_desired(sampled_tcrs.j_gene.value_counts(), j_desired)
        len_desired = _update_n_desired(sampled_tcrs.junction_aa.str.len().value_counts(), len_desired)
        total -= chunksize
        if total <= 0:
            break
    cols = ['junction','junction_aa','v_call','v_gene','j_call','j_gene']
    return pd.concat(background).reset_index(drop=True)[cols]

def compare_frequencies(seq_source, seq_bg_matched):

    seq_source['v_gene'] = seq_source.v_call.apply(lambda x: x.split('*')[0])
    seq_source['j_gene'] = seq_source.j_call.apply(lambda x: x.split('*')[0])
    seq_bg_matched['v_gene'] = seq_bg_matched.v_call.apply(lambda x: x.split('*')[0])
    seq_bg_matched['j_gene'] = seq_bg_matched.j_call.apply(lambda x: x.split('*')[0])

    vcounts_source = (seq_source.v_gene.value_counts() / len(seq_source))
    vcounts_matched = (seq_bg_matched.v_gene.value_counts() / len(seq_bg_matched))
    vcounts = pd.concat([vcounts_source, vcounts_matched], axis=1).reset_index()
    vcounts.columns = ['v', 'source', 'matched']
    vcounts = pd.melt(vcounts, id_vars=['v'], value_vars=['source', 'matched'])

    jcounts_source = (seq_source.j_gene.value_counts() / len(seq_source))
    jcounts_matched = (seq_bg_matched.j_gene.value_counts() / len(seq_bg_matched))
    jcounts = pd.concat([jcounts_source, jcounts_matched], axis=1).reset_index()
    jcounts.columns = ['j', 'source', 'matched']
    jcounts = pd.melt(jcounts, id_vars=['j'], value_vars=['source', 'matched'])

    counts_source = seq_source.junction_aa.str.len().value_counts().sort_index() / len(seq_source)
    counts_matched = seq_bg_matched.junction_aa.str.len().value_counts().sort_index() / len(seq_bg_matched)
    len_counts = pd.concat([counts_source, counts_matched], axis=1).reset_index()
    len_counts.columns = ['length', 'source', 'matched']
    len_counts = pd.melt(len_counts, id_vars=['length'], value_vars=['source', 'matched'])

    return vcounts, jcounts, len_counts

    # def parse_junctions_for_background_resampling(df, organism, chain):
    #     ''' 
    #     Setup for the "resample_background_tcrs_v4" function by parsing
    #     the V(D)J junctions in the foreground tcr set

    #     returns a dataframe with info
    #     '''
    #     from tcrdist.tcr_sampler import parse_tcr_junctions

    #     # tcrdist parsing function expects paired tcrs as list of tuples of tuples
    #     cols = ['v_call', 'j_call', 'junction_aa', 'junction']
    #     tcr_tuples = df[cols].itertuples(name=None, index=None)
    #     if chain == 'A':
    #         tcr_tuples = zip(tcr_tuples, itertools.repeat(None))
    #     else:
    #         tcr_tuples = zip(itertools.repeat(None), tcr_tuples)

    #     junctions = parse_tcr_junctions(organism, list(tcr_tuples))
    #     #junctions = add_vdj_splits_info_to_junctions(junctions)
    #     return junctions

class SyntheticBackground():
    def __init__(
        self, 
        repertoire, 
        factor=10, 
        chain:str='B', 
        organism:str='human',
        v_column:str='v_call',
        j_column:str='j_call',
        cdr3nt_column:str='junction',
        cdr3aa_column:str='junction_aa', 
        ncpus:int=1,
        verbose=False
        ):
        '''
        Class for creating synthetic backgrounds that match certain properties of the
        input repertoire.
        '''
        self.repertoire = repertoire
        self.factor = factor
        self.chain = chain
        self.organism = organism
        self.junctions = None
        self.size = len(self.repertoire)
        self.n = self.size * self.factor
        self._setup_gene_ref()
        self._setup_olga()
        # Prepare gene columns
        self.v_column = v_column
        self.j_column = j_column
        self.cdr3nt_column = cdr3nt_column
        self.cdr3aa_column = cdr3aa_column
        self._add_v_gene_column()
        self._add_j_gene_column()
        self.v_genes = self.repertoire.v_gene.unique()

        # Configure number of threads
        if ncpus == -1:
            self.ncpus = mp.cpu_count()
        elif ncpus >= mp.cpu_count():
            self.ncpus = mp.cpu_count()
        else:
            assert ncpus > 0, 'Number of CPUs must be greater than 0.'
            self.ncpus = ncpus

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

    def _setup_olga(self):
        '''
        Configure OLGA marginals and model parameters.
        '''
        params_file_name = join(MODELS,'model_params.txt')
        marginals_file_name = join(MODELS,'model_marginals.txt')
        V_anchor_pos_file = join(MODELS,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = join(MODELS,'J_gene_CDR3_anchors.csv')
    
        self.genomic_data = load_model.GenomicDataVDJ()
        self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

        self.generative_model = load_model.GenerativeModelVDJ()
        self.generative_model.load_and_process_igor_model(marginals_file_name)

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

    def generate_sequence_vj(self, v=None, j=None):
        '''
        Generate a VDJ sequence using the OLGA human TRB model, specifying
        V and/or J gene selection.
        '''
        v_id = v
        j_id = j
        seq_gen_model = olga_directed.SequenceGenerationVDJ(self.generative_model, self.genomic_data)
        if v is not None:
            if not '*' in v:
                v = random.choice(self.v_gene_ref[v])
            v_id = self.v_allele_ref[v]
        if j is not None:
            if not '*' in j:
                j = random.choice(self.j_gene_ref[j])
            j_id = self.j_allele_ref[j]
        recomb = None
        while recomb is None:
            recomb = seq_gen_model.gen_rnd_prod_CDR3(V=v_id, J=j_id)
        aaseq, v_id, j_id = recomb[1:4]
        v_allele = self.genomic_data.genV[v_id][0]
        j_allele = self.genomic_data.genJ[j_id][0]
        return aaseq, v_allele, j_allele

    def gene_length_distribution(self, gene:str, feature:str='v_gene'):
        '''
        Get CDR3 length distribution for all sequences with a specific V or J gene.

        Parameters
        ----------
        gene: str
            Name of the gene.
        feature: str
            Name of the column containing the V or J genes. Default is v_gene.
        '''
        v_len_counts = self.repertoire[self.repertoire[feature]==gene].junction_aa.str.len().value_counts()
        return (v_len_counts / self.size * self.n).astype(int)

    def bin_sequences(self, v_gene:str=None, j_gene:str=None):
        '''
        AR sampling method for matching the CDR3 length distribution 
        for a specific V or J gene.

        Parameters
        ----------
        v_gene: str
            V gene.
        j_gene: str
            J gene.
        '''
        n_required = self.gene_length_distribution(gene=v_gene)
        res = []
        while sum(n_required) > 0:
            aa, v, j = self.generate_sequence_vj(v=v_gene, j=j_gene)
            if len(aa) in n_required.index:
                if n_required.loc[len(aa)] > 0:
                    res.append((aa,v,j))
                    n_required.loc[len(aa)] -= 1
        return pd.DataFrame(res, columns=['junction_aa', 'v_call', 'j_call'])

    def match_per_v_gene(self):
        '''
        Generate a synthetic repertoire that matches the foreground CDR3 length distributions 
        per V gene.
        '''
        if self.ncpus == 1:
            sampled = [self.bin_sequences(v_gene=v) for v in self.v_genes]
        else:
            with mp.Pool(self.ncpus) as pool:
                sampled = parmap.map(
                    self.bin_sequences,
                    self.v_genes,
                    np.random.seed(),
                    pm_parallel=True,
                    pm_pool=pool
                )
        return pd.concat(sampled)

    def match_per_j_gene(self):
        '''
        Generate a synthetic repertoire that matches the foreground CDR3 length distributions 
        per J gene.
        '''
        if self.ncpus == 1:
            sampled = [self.bin_sequences(j_gene=j) for j in self.j_genes]
        else:
            with mp.Pool(self.ncpus) as pool:
                sampled = parmap.map(
                    self.bin_sequences,
                    self.j_genes,
                    np.random.seed(),
                    pm_parallel=True,
                    pm_pool=pool
                )
        return pd.concat(sampled)

    def full_match(self):

        # Initialize probabilities
        sampling_probabilities_v = self.repertoire.v_gene.value_counts() / self.size
        v_necessary = sampling_probabilities_v * self.n
        sampling_probabilities_j = self.repertoire.j_gene.value_counts() / self.size
        j_necessary = sampling_probabilities_j * self.n
        sampling_probabilities_len = self.repertoire.junction_aa.str.len().value_counts() / self.size
        len_necessary = sampling_probabilities_len * self.n
        for i in range(41):
            if i not in len_necessary.index:
                len_necessary.loc[i] = 0

        res = []
        nsample = self.n

        for i in range(self.n):
            recombination = None
            while recombination is None:
                vchoice = random.choices(v_necessary.keys(), v_necessary.values, k=1)[0]
                jchoice = random.choices(j_necessary.keys(), j_necessary.values, k=1)[0]
                recombination = self.generate_sequence_vj(v=vchoice, j=jchoice)
                # recombination = olga_sequence(vgene=vchoice, jgene=jchoice, vref=vref, jref=jref, seq_gen_model=seq_gen_model)
            aaseq, v, j = recombination

            while len_necessary[len(aaseq)] <= 0:
                recombination = self.generate_sequence_vj(v=vchoice, j=jchoice)
                # recombination = olga_sequence(vgene=vchoice, jgene=jchoice, vref=vref, jref=jref, seq_gen_model=seq_gen_model)
                if recombination is not None:
                    aaseq, v, j = recombination
                else:
                    pass

            res.append(recombination)
            len_necessary[len(aaseq)] -= 1
            if len_necessary[len(aaseq)] <= 0:
                len_necessary[len(aaseq)] = 0

            v_necessary[vchoice] -= 1
            if v_necessary[vchoice] <= 0:
                v_necessary[vchoice] = 0
            j_necessary[jchoice] -= - 1
            if j_necessary[jchoice] <= 0:
                j_necessary[jchoice] = 0

            nsample -= 1
            if nsample <= 0:
                break
            
            # Update 'probabilities'
            sampling_probabilities_v = self.repertoire.v_gene.value_counts() / nsample
            sampling_probabilities_j = self.repertoire.j_gene.value_counts() / nsample

        cols = ['junction_aa','v_call','j_call']
        return pd.DataFrame(res, columns=cols)

    def _parse_junctions(self):
        '''
        setup for the "resample_background_tcrs" function by parsing
        the V(D)J junctions in the foreground tcr set

        returns a dataframe with info
        '''
        from tcrdist_old.tcr_sampler import parse_tcr_junctions

        # tcrdist parsing function expects paired tcrs as list of tuples of tuples
        tcr_columns = [
            self.v_column, 
            self.j_column, 
            self.cdr3aa_column, 
            self.cdr3nt_column
            ]
        self.repertoire[self.cdr3nt_column] = self.repertoire[self.cdr3nt_column].str.lower()
        tcr_tuples = self.repertoire[tcr_columns].itertuples(name=None, index=None)
        if self.chain == 'A':
            tcr_tuples = zip(tcr_tuples, itertools.repeat(None))
        else:
            tcr_tuples = zip(itertools.repeat(None), tcr_tuples)

        self.junctions = parse_tcr_junctions(self.organism, list(tcr_tuples))
        #junctions = add_vdj_splits_info_to_junctions(junctions)
        return self.junctions

    def resample_background_tcrs(
        self,
        preserve_vj_pairings=False, # consider setting True for alpha chain
        return_nucseq_srcs=False, # for debugging
        verbose=False
        ):
        '''
        Resample shuffled background sequences, number will be equal to size of
        foreground repertore, ie junctions.shape[0]

        junctions is a dataframe with information about the V(D)J junctions in the
        foreground tcrs. Created by the function above this one,
            "parse_junctions_for_background_resampling"

        returns a list of tuples [(v,j,cdr3aa,cdr3nt), ...] of length = junctions.shape[0]
        '''
        from tcrdist_old.tcr_sampler import resample_shuffled_tcr_chains

        if self.junctions is None:
            self._parse_junctions()

        multiplier = 3 # so we have enough to match distributions
        bg_tcr_tuples, src_junction_indices = resample_shuffled_tcr_chains(
            self.organism, multiplier * self.junctions.shape[0], self.chain, self.junctions,
            preserve_vj_pairings=preserve_vj_pairings,
            return_src_junction_indices=True
            )

        fg_nucseq_srcs = list(self.junctions[f'cdr3{self.chain.lower()}_nucseq_src'])

        resamples = []
        for tcr, inds in zip(bg_tcr_tuples, src_junction_indices):
            if len(resamples)%500000==0:
                if self.verbose:
                    print(f'{SyntheticBackground.resample_background_tcrs.__name__}: build nucseq_srclist', len(resamples),
                        len(bg_tcr_tuples))
            v,j,cdr3aa,cdr3nt = tcr
            v_nucseq = fg_nucseq_srcs[inds[0]]
            j_nucseq = fg_nucseq_srcs[inds[1]]
            nucseq_src = v_nucseq[:inds[2]] + j_nucseq[inds[2]:]
            assert len(tcr[3]) == len(nucseq_src)
            resamples.append((v, j, cdr3aa, cdr3nt, nucseq_src))


        # try to match lengths first
        fg_lencounts = Counter(len(x) for x in self.junctions['cdr3'+self.chain.lower()])

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

        if self.chain == 'B':
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
                    print(f'{SyntheticBackground.resample_background_tcrs.__name__} dont have enough Ns:',
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


    def shuffle_repertoire(self):
        bg = pd.concat([pd.DataFrame(self.resample_background_tcrs()) for i in range(self.factor)])
        bg.columns = [self.v_column, self.j_column, self.cdr3aa_column, self.cdr3nt_column]
        return bg


def get_gene_id(gene, ref):
    gene = gene.split('*')[0]
    assert gene in list(ref.gene), f'Unknown gene: {vgene}.'
    return int(random.choice(ref[ref['gene']==gene].index))

def olga_sequence(seq_gen_model, vgene=None, jgene=None, vref=None, jref=None):
    if vgene is not None:
        vid = int(get_gene_id(vgene, vref))
    else:
        vid = vgene
    if jgene is not None:
        jid = int(get_gene_id(jgene, jref))
    else:
        jid = jgene
    # recombination = None
    # while recombination is None:
    recombination = seq_gen_model.gen_rnd_prod_CDR3(V=vid, J=jid)
    if recombination is None:
        return None
    else:
        cdr3_nt_out = recombination[0]
        cdr3_aa_out = recombination[1]
        v_out = vref.loc[recombination[2]].v_allele
        j_out = jref.loc[recombination[3]].j_allele
        return (cdr3_nt_out, cdr3_aa_out, v_out, j_out)

def directed_sampling(df, nsample):

    from raptcr.constants.modules.olga import olga_directed

    genomic_data, generative_model = _setup_olga_models()
    seq_gen_model = olga_directed.SequenceGenerationVDJ(generative_model, genomic_data)
    
    # Setup V gene reference
    vref = pd.DataFrame(genomic_data.genV, columns=['v_allele', 'ntseq1', 'ntseq2'])
    vref['gene'] = vref.v_allele.apply(lambda x: x.split('*')[0])
    v_anchor = pd.read_csv('./raptcr/constants/modules/olga/default_models/human_T_beta/V_gene_CDR3_anchors.csv')
    vref = vref[vref.v_allele.isin(v_anchor[v_anchor.function=='F'].gene)]
    # Setup J gene reference
    jref = pd.DataFrame(genomic_data.genJ, columns=['j_allele', 'ntseq1', 'ntseq2'])
    jref['gene'] = jref.j_allele.apply(lambda x: x.split('*')[0])
    j_anchor = pd.read_csv('./raptcr/constants/modules/olga/default_models/human_T_beta/J_gene_CDR3_anchors.csv')
    jref = jref[jref.j_allele.isin(j_anchor[j_anchor.function=='F'].gene)]

    # Initialize probabilities
    sampling_probabilities_v = df.v_call.value_counts() / len(df)
    v_necessary = sampling_probabilities_v * nsample
    sampling_probabilities_j = df.j_call.value_counts() / len(df)
    j_necessary = sampling_probabilities_j * nsample
    sampling_probabilities_len = df.junction_aa.str.len().value_counts() / len(df)
    len_necessary = sampling_probabilities_len * nsample
    for i in range(41):
        if i not in len_necessary.index:
            len_necessary.loc[i] = 0

    res = []

    for i in range(nsample):
        recombination = None
        while recombination is None:
            vchoice = random.choices(v_necessary.keys(), v_necessary.values, k=1)[0]
            jchoice = random.choices(j_necessary.keys(), j_necessary.values, k=1)[0]
            recombination = olga_sequence(vgene=vchoice, jgene=jchoice, vref=vref, jref=jref, seq_gen_model=seq_gen_model)
        nucseq, aaseq, v, j = recombination

        while len_necessary[len(aaseq)] <= 0:
            recombination = olga_sequence(vgene=vchoice, jgene=jchoice, vref=vref, jref=jref, seq_gen_model=seq_gen_model)
            if recombination is not None:
                nucseq, aaseq, v, j = recombination
            else:
                pass

        res.append(recombination)
        len_necessary[len(aaseq)] -= 1
        if len_necessary[len(aaseq)] <= 0:
            len_necessary[len(aaseq)] = 0

        v_necessary[vchoice] -= 1
        if v_necessary[vchoice] <= 0:
            v_necessary[vchoice] = 0
        j_necessary[jchoice] -= - 1
        if j_necessary[jchoice] <= 0:
            j_necessary[jchoice] = 0

        nsample -= 1
        if nsample <= 0:
            break
        
        # Update 'probabilities'
        sampling_probabilities_v = df.v_call.value_counts() / nsample
        sampling_probabilities_j = df.j_call.value_counts() / nsample

    cols = ['junction','junction_aa','v_call','j_call']
    return pd.DataFrame(res, columns=cols)
