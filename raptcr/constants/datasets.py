import pandas as pd
import random

FILENAME = './raptcr/datasets/big_background.tsv'

def sample_cdr3s(s:int, background:str=FILENAME, aa_col:str='junction_aa') -> list:
    '''
    Randomly sample s number of cdr3s from the background distribution.
    '''
    n = sum(1 for line in open(FILENAME)) - 1 # number of records in file (excludes header)
    skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
    return pd.read_csv(background, sep='\t', skiprows=skip)[aa_col].to_list()

def sample_tcrs(s:int, background:str=FILENAME, cols:list=['junction_aa','v_call','j_call']):
    n = sum(1 for line in open(FILENAME)) - 1 # number of records in file (excludes header)
    skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
    return pd.read_csv(background, sep='\t', skiprows=skip)[cols]

example =  list(pd.read_csv('./raptcr/datasets/example_repertoire.tsv', sep='\t').junction_aa.unique())
example_tcr = pd.read_csv('./raptcr/datasets/example_repertoire.tsv', sep='\t')