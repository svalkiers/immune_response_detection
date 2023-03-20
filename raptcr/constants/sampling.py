import pandas as pd
import numpy as np
import math
import multiprocessing
import parmap
import random

from .datasets import sample_tcrs
from .modules.OLGA.olga import load_model, sequence_generation

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

    params_file_name = './raptcr/constants/modules/OLGA/olga/default_models/human_T_beta/model_params.txt'
    marginals_file_name = './raptcr/constants/modules/OLGA/olga/default_models/human_T_beta/model_marginals.txt'
    V_anchor_pos_file ='./raptcr/constants/modules/OLGA/olga/default_models/human_T_beta/V_gene_CDR3_anchors.csv'
    J_anchor_pos_file = './raptcr/constants/modules/OLGA/olga/default_models/human_T_beta/J_gene_CDR3_anchors.csv'

    genomic_data = load_model.GenomicDataVDJ()
    genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

    generative_model = load_model.GenerativeModelVDJ()
    generative_model.load_and_process_igor_model(marginals_file_name)
    
    return genomic_data, generative_model

def generate_olga_sequences(n, genomic_data, generative_model):
    seq_gen_model = sequence_generation.SequenceGenerationVDJ(generative_model, genomic_data)
    colnames = ['junction','junction_aa','v_call','j_call']
    df = pd.DataFrame([seq_gen_model.gen_rnd_prod_CDR3() for i in range(n)], columns=colnames)
    df['v_call'] = df['v_call'].apply(lambda x: genomic_data.genV[x][0])
    df['j_call'] = df['j_call'].apply(lambda x: genomic_data.genJ[x][0])
    df['v_gene'] = df['v_call'].apply(lambda x: x.split('*')[0])
    df['j_gene'] = df['j_call'].apply(lambda x: x.split('*')[0])
    return df

def generate_olga_sequences_multi(total, chunks, genomic_data, generative_model, ncpus=1):
    with multiprocessing.Pool(ncpus) as pool:
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
        while sum(weights) <= 0:
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