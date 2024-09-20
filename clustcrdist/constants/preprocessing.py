import pandas as pd
from os.path import dirname, abspath, join
from .base import GAPCHAR

ROOT = dirname(dirname(dirname(abspath(__file__))))
DATA = join(ROOT, 'clustcrdist/constants/data')

IMGT = pd.read_csv(join(DATA,'imgt_reference.tsv'), sep='\t')
mapping = pd.read_csv(join(DATA, 'adaptive_imgt_mapping.csv'))

adaptive_to_imgt_human = mapping.loc[mapping['species'] == 'human'].set_index('adaptive')['imgt'].fillna('NA').to_dict()
adaptive_to_imgt_mouse = mapping.loc[mapping['species'] == 'mouse'].set_index('adaptive')['imgt'].fillna('NA').to_dict()

v_fam_freq = pd.read_csv(join(DATA,"adaptive_v_fam_to_imgt_gene.txt"), sep="\t")
adaptive_vfam_mapping = dict(zip(v_fam_freq.adaptive_v_family, v_fam_freq.imgt_v_allele))

def vgene_to_cdr():
    '''
    Load V gene reference file that contains mapping information to extract 
    CDR1 and CDR2 information based on the V gene allele annotation.
    '''
    return pd.read_csv(join(DATA,'vgene_to_cdr.txt'), sep='\t')

def get_gene_reference():
    return pd.read_csv(join(DATA,'combo_xcr.tsv'), sep='\t')

def add_cdr_columns(df:pd.DataFrame, vcol:str='v_call') -> pd.DataFrame:
    '''
    Adds CDR1 and CDR2 information to TCR sequences based on V gene allele annotation.
    '''
    vgene = vgene_to_cdr()
    df['cdr1_b_aa'] = df[vcol].map(dict(zip(vgene['id'],vgene['cdr1'])))
    df['cdr2_b_aa'] = df[vcol].map(dict(zip(vgene['id'],vgene['cdr2'])))
    return df

def to_tcrdist3_format(df:pd.DataFrame, vgenecol:str='v_call', jgenecol:str='j_call', cdr3col:str='junction_aa'):
    return df.rename(columns={
        vgenecol:'v_b_gene',
        jgenecol:'j_b_gene',
        cdr3col:'cdr3_b_aa'
    })

def format_organism(organism):
    '''
    Correctly format the input organism.
    '''
    mapping = {
        'human':'human',
        'homo':'human',
        'homo_sapiens':'human',
        'homosapiens':'human',
        'sapiens':'human',
        'mouse':'mouse',
        'mus':'mouse',
        'musmusculus':'mouse',
        'mus_musculus':'mouse',
        'musculus':'mouse'
        }
    assert organism.lower() in mapping, f"Unknown organism: {organism}. Please select human or mouse."
    return mapping[organism.lower()]

def format_chain(chain):
    '''
    Correctly format the input TCR chain.
    '''
    mapping = {
        'b':'B',
        'beta':'B',
        'trb':'B',
        'trbeta':'B',
        'tcrb':'B',
        'tcrbeta':'B',
        'tcr_beta':'B',
        'a':'A',
        'alpha':'A',
        'tra':'A',
        'tralpha':'A',
        'tcra':'A',
        'tcralpha':'A',
        'tcr_alpha':'A',
        'ab':'AB',
        'alphabeta':'AB',
        'alpha_beta':'AB',
        'g':'G',
        'gamma':'G',
        'trg':'G',
        'trgamma':'G',
        'tcrg':'G',
        'tcrgamma':'G',
        'tcr_gamma':'G',
        'd':'D',
        'delta':'D',
        'trd':'D',
        'trdelta':'D',
        'tcrd':'D',
        'tcrdelta':'D',
        'tcr_delta':'D',
        'gd':'GD',
        'gammadelta':'GD',
        'gamma_delta':'GD',
    }
    assert chain.lower() in mapping, f"Unknown chain: {chain}. Please select A, B or AB."
    return mapping[chain.lower()]

def setup_gene_cdr_strings(organism:str='human', chain:str='B'):
    ''' 
    Returns dict mapping vgene names to concatenated cdr1-cdr2-cdr2.5 strings
    columns without any sequence variation (e.g. all gaps) are removed
    '''
    # Make sure organism and chain input is correctly formatted
    organism = format_organism(organism)
    chain = format_chain(chain)
    # Get CDR information from gene reference file
    all_genes_df = get_gene_reference()
    all_genes_df = all_genes_df[(all_genes_df.organism==organism)&
                                (all_genes_df.chain.isin(list(chain)))&
                                (all_genes_df.region=='V')]
    assert all_genes_df.id.value_counts().max()==1
    all_genes_df.set_index('id', inplace=True)
    all_genes_df['cdrs'] = all_genes_df.cdrs.str.split(';')
    vgenes = list(all_genes_df.index)
    gene_cdr_strings = {x:'' for x in vgenes}
    oldgap = '.' # gap character in the all_genes dict
    # print(gene_cdr_strings)
    for icdr in range(3):
        cdrs = all_genes_df.cdrs.str.get(icdr).str.replace(oldgap,GAPCHAR,regex=False)
        L = len(cdrs[0])
        for i in reversed(range(L)):
            # print(L)
            col = set(x[i] for x in cdrs)
            if len(col) == 1: # no variation
                cdrs = [x[:i]+x[i+1:] for x in cdrs]
        for g, cdr in zip(vgenes, cdrs):
            gene_cdr_strings[g] += cdr
    return gene_cdr_strings

def detect_vgene_col(df):
    pattern = r'TR[A,B,G,D]V'
    matching_columns = [column for column in df.columns if df[column].astype(str).str.contains(pattern, regex=True).all()]
    if len(matching_columns) == 1:
        print("Autodetected V gene column:", matching_columns[0])
        return matching_columns[0]
    elif len(matching_columns) == 0:
        raise ValueError("No V gene column detected. If your dataframe contains V gene information, please make sure it is IMGT-formatted.")
    else:
        raise ValueError("Multiple V gene columns detected. Please specify the correct column name.")

def detect_cdr3_col(df):
    pattern = r'^[CW].*[FWC]$'
    matching_columns = [column for column in df.columns if df[column].astype(str).str.contains(pattern, regex=True).all()]
    if len(matching_columns) == 1:
        print("Autodetected CDR3AA column:", matching_columns[0])
        return matching_columns[0]
    elif len(matching_columns) == 0:
        raise ValueError("No CDR3 column detected. Please make sure your dataframe contains a column with CDR3 sequences or specify the column name.")
    else:
        raise ValueError("Multiple CDR3 columns detected. Please specify the correct column name.")