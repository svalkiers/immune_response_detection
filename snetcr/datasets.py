import pandas as pd

from os.path import join, dirname, abspath

DIR = dirname(abspath(__file__))

def load_test():
    return pd.read_table(join(DIR, 'data/test.tsv'))
