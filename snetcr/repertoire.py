import pandas as pd

from .constants.preprocessing import IMGT
from .constants.parsing import (
    _is_cdr3,
    format_junction,
    )

class Repertoire():
    '''
    Repertoire formatter.
    Data must be in the AIRR format.
    '''
    
    def __init__(self,data):
        self.data = data

    def filter_and_format(
            self,
            cdr3aa_col = 'junction_aa',
            cdr3nt_col = 'junction',
            vgene_col = 'v_call',
            jgene_col = 'j_call',
            remove_nonfunctional=True
            ):
        
        '''
        Filter and format the data. Filtering includes:
        - Removing ambiguous CDR3 amino acid sequences
        - Removing ORF and non-functional genes
        - Parsing junction (CDR3 nucleotide sequence)
        '''

        col_remap = {
            cdr3nt_col:"junction",
            cdr3aa_col:"junction_aa",
            vgene_col:"v_call",
            jgene_col:"j_call",
            }
        
        self.data = self.data.rename(columns=col_remap)

        # Remove ambiguous CDR3 amino acid sequences
        print("Remove ambiguous CDR3 amino acid sequences")
        self.data = self.data[self.data.junction_aa.apply(lambda cdr3: _is_cdr3(cdr3))]
        
        # Remove ORF and non-functional genes
        print("Parsing V/J genes")
        if remove_nonfunctional:
            functional = IMGT[IMGT['fct'].isin(['F','(F)','[F]'])]
            self.data = self.data[self.data.v_call.isin(functional.imgt_allele_name)]
        else:
            # make sure all V genes have IMGT annotation
            self.data = self.data[self.data.v_call.isin(IMGT.imgt_allele_name)]
        
        # Parse junction (CDR3 nucleotide sequence)
        print("Parsing CDR3 nucleotide sequence")
        self.data = format_junction(self.data)
        self.data = self.data.dropna(subset=["junction_aa","junction","v_call","j_call"])

    def airr_to_tcrdist_paired(self):
        '''
        Convert AIRR data to paired tcrdist format.
        '''
        # Splitting the dataframe into TRA and TRB
        alpha_data = self.data[self.data['locus'] == 'TRA'].copy()
        beta_data = self.data[self.data['locus'] == 'TRB'].copy()
        # Renaming columns for merging
        alpha_data = alpha_data.rename(columns={'v_call': 'va', 'j_call': 'ja', 'junction_aa': 'cdr3a', 'junction': 'cdr3a_nucseq'})
        beta_data = beta_data.rename(columns={'v_call': 'vb', 'j_call': 'jb', 'junction_aa': 'cdr3b', 'junction': 'cdr3b_nucseq'})
        # alpha_data = alpha_data.drop(columns=['locus'])
        # beta_data = beta_data.drop(columns=['locus'])
        
        possible_id_cols = [['cell_id', 'clone_id'], ['cell_id'], ['clone_id']]
        id_cols = next((cols for cols in possible_id_cols if all(col in self.data.columns for col in cols)), None)
        if id_cols is None:
            raise ValueError("No valid id columns found in the dataframe. Paired data must include clone_id and/or cell_id.")
        
        alpha_cols = ['va', 'ja', 'cdr3a', 'cdr3a_nucseq'] + id_cols
        beta_cols = ['vb', 'jb', 'cdr3b', 'cdr3b_nucseq'] + id_cols
        alpha_data = alpha_data[alpha_cols]
        beta_data = beta_data[beta_cols]
        
        merged = pd.merge(alpha_data, beta_data, on=id_cols, how='inner')
        merged = merged.dropna()
        return merged