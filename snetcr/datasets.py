import pandas as pd

from os.path import join, dirname, abspath

DIR = dirname(abspath(__file__))

def load_test(column_type='single'):
    if column_type == 'single':
        return pd.read_table(join(DIR, 'data/test_single.tsv'))
    elif column_type == 'paired':
        return pd.read_table(join(DIR, 'data/test_paired.tsv'))
