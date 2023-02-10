import pandas as pd

def get_vgene_reference():
    '''
    Load V gene reference file that contains mapping information to extract 
    CDR1 and CDR2 information based on the V gene allele annotation.
    '''
    return pd.read_csv('./raptcr/constants/vgene_to_cdr.txt', sep='\t')

def add_cdr_columns(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Adds CDR1 and CDR2 information to TCR sequences based on V gene allele annotation.
    '''
    vgene = get_vgene_reference()
    df['cdr1'] = df.v_call.map(dict(zip(vgene['id'],vgene['cdr1'])))
    df['cdr2'] = df.v_call.map(dict(zip(vgene['id'],vgene['cdr2'])))
    return df

def to_tcrdist3_format(df:pd.DataFrame, vgenecol:str='v_call', jgenecol:str='j_call', cdr3col:str='junction_aa'):
    return df.rename(columns={
        vgenecol:'v_b_gene',
        jgenecol:'j_b_gene',
        cdr3col:'cdr3_b_aa'
    })