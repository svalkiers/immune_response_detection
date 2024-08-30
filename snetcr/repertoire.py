import pandas as pd
import numpy as np

from .constants.preprocessing import IMGT, adaptive_to_imgt_human, adaptive_to_imgt_mouse
from .constants.parsing import (
    _is_cdr3,
    format_junction,
    )

class Repertoire():
    '''
    Repertoire formatter.
    '''
    
    def __init__(self,data,organism='human'):
        self.data = data
        self.organism = organism

    def filter_and_format_single(
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
            
        if self.data.v_call.str.contains('TCRBV').any():
            print('Detected Adaptive format')
            self.data = self.data[self.data.frame_type == 'In']
            self.data["v_imgt"] = self.data.v_call.map(adaptive_to_imgt_human)
            self.data["j_call"] = self.data.j_call.map(adaptive_to_imgt_human)
            # Correct the V allele if available else use IMGT mapping
            self.data = self.data.dropna(subset=["v_imgt","junction_aa","junction","j_call"])
            self.data["v_call"] = [self.data.v_imgt.iloc[n].split("*")[0]+f"*0{int(i)}" if not np.isnan(i) else self.data.v_imgt.iloc[n] for n,i in enumerate(self.data.v_allele)]
            self.data = self.data.drop(columns=["v_imgt"])

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

        return self.data
    
    def filter_and_format_paired(
            self,
            cdr3aa_a_col = 'cdr3a',
            cdr3nt_a_col = 'cdr3a_nucseq',
            vgene_a_col = 'va',
            jgene_a_col = 'ja',
            cdr3aa_b_col = 'cdr3b',
            cdr3nt_b_col = 'cdr3b_nucseq',
            vgene_b_col = 'vb',
            jgene_b_col = 'jb',
            remove_nonfunctional=True
            ):
        
        col_remap = {
            cdr3nt_a_col:"cdr3a",
            cdr3aa_a_col:"cdr3a_nucseq",
            vgene_a_col:"va",
            jgene_a_col:"ja",
            cdr3nt_b_col:"cdr3b",
            cdr3aa_b_col:"cdr3b_nucseq",
            vgene_b_col:"vb",
            jgene_b_col:"jb",
            }

        # Remove ambiguous CDR3 amino acid sequences
        print("Remove ambiguous CDR3 amino acid sequences")
        self.data = self.data[self.data.cdr3a.apply(lambda cdr3: _is_cdr3(cdr3)) &
                              self.data.cdr3b.apply(lambda cdr3: _is_cdr3(cdr3))]

        # Remove ORF and non-functional genes
        print("Parsing V/J genes")
        if remove_nonfunctional:
            functional = IMGT[IMGT['fct'].isin(['F','(F)','[F]'])]
            self.data = self.data[self.data.va.isin(functional.imgt_allele_name) & 
                                  self.data.vb.isin(functional.imgt_allele_name)]
        else:
            # make sure all V genes have IMGT annotation
            self.data = self.data[self.data.va.isin(IMGT.imgt_allele_name) & 
                                  self.data.vb.isin(IMGT.imgt_allele_name)]
        
        tcra = self.data[['cdr3a','cdr3a_nucseq','va','ja']]
        tcrb = self.data[['cdr3b','cdr3b_nucseq','vb','jb']]
        tcra.columns = ['junction_aa','junction','v_call','j_call']
        tcrb.columns = ['junction_aa','junction','v_call','j_call']
        tcra = format_junction(tcra)
        tcrb = format_junction(tcrb)
        self.data['cdr3a_nucseq'] = tcra['junction']
        self.data['cdr3b_nucseq'] = tcrb['junction']
        self.data = self.data.dropna(subset=["cdr3a","cdr3b","cdr3a_nucseq","cdr3b_nucseq","va","ja","vb","jb"])

        # # Apply format_junction directly to the relevant columns
        # self.data[['cdr3a_nucseq', 'cdr3b_nucseq']] = self.data[['cdr3a', 'cdr3a_nucseq', 'va', 'ja', 'cdr3b', 'cdr3b_nucseq', 'vb', 'jb']].apply(
        #     lambda x: format_junction(x[['cdr3a', 'cdr3a_nucseq', 'va', 'ja']],'cdr3a','cdr3a_nucseq','va','ja')['junction']
        #     if x.name == 'cdr3a_nucseq' else format_junction(x[['cdr3b', 'cdr3b_nucseq', 'vb', 'jb']],'cdr3b', 'cdr3b_nucseq', 'vb', 'jb')['junction'], axis=1
        # )

        return self.data


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