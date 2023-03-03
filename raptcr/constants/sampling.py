import pandas as pd
import numpy as np

from .datasets import sample_tcrs

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
            background = sample_tcrs(n)
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

    # Sample V and J genes according to gene family frequencies in the foreground
    vgenes = pd.concat([background[background.vfam==v].v_call.sample(vfam_counts[v], replace=True) for v in vfam_counts])
    jgenes = pd.concat([background[background.jfam==j][['j_call','junction_aa']].sample(jfam_counts[j], replace=True) for j in jfam_counts])
    
    if actual_n < n:
        vgenes = vgenes.sample(actual_n)
        jgenes = jgenes.sample(actual_n)
    else:
        pass

    return pd.concat([vgenes.reset_index(drop=True), jgenes.reset_index(drop=True)], axis=1).dropna()