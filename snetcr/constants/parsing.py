import os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

# from .constants.modules import tcrdist
from .modules.tcrdist.all_genes import all_genes # for recognized genes
from .modules.tcrdist.translation import get_translation
from .modules.tcrdist.tcr_sampler import get_j_cdr3_nucseq
from .preprocessing import IMGT, adaptive_to_imgt_human, adaptive_vfam_mapping

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
_aminoacids_set = set(aminoacids)

def _is_aaseq(seq:str):
    """
    Check if string contains non-amino acid characters.
    Returns True if string only contains standard amino acid characters.
    """
    try:
        return all(c in _aminoacids_set for c in seq)
    except TypeError:
        return False

def _is_cdr3(seq:str):
    """
    Checks if string is a valid CDR3 amino acid sequence,
    according to the following defenitions:
        - First amino acid character is C.
        - Last amino acid is F or W.
        - Sequence exclusively contains valid amino acid characters.
    """
    try:
        return (_is_aaseq(seq)
            and (seq[0] == 'C')
            and (seq[-1] in ['F', 'W', 'C'])
            and (len(seq) <= 30)
            and (len(seq) >= 4))
    # Exclude non-string type input
    except TypeError:
        return False

def capture_nucseq(tcr, organism='human'):

    vgene, cdr3, rearrangement, jgene = tcr

    # figure out the cdr3 nucseq
    cdr3_nucseq = None
    jgene_cdr3_nucseq = get_j_cdr3_nucseq(organism,jgene).upper() # added .upper()

    for offset in range(3):
        protseq = get_translation( rearrangement, '+{}'.format(offset+1) )
        for ctrim in range(1,4):
            if cdr3[:-ctrim] in protseq:
                start = offset + 3*(protseq.index(cdr3[:-ctrim]))
                length = 3*(len(cdr3)-ctrim)
                cdr3_nucseq = rearrangement[ start : start + length ]
                if ctrim:
                    cdr3_nucseq += jgene_cdr3_nucseq[-3*ctrim:]
                if cdr3 != get_translation( cdr3_nucseq, '+1' ):
                    cdr3_nucseq = ''
                break
        if cdr3_nucseq=='': # failure signal
            cdr3_nucseq = None
            break

    if cdr3_nucseq is None:
        print('parse cdr3_nucseq failed:',tcr)
        return np.nan
    else:
        return cdr3_nucseq

def get_tcr(row):
    return (row.v_call, row.junction_aa, row.junction, row.j_call)

def format_junction(data):
    n_pre = data.shape[0] # number of TCRs in data before parsing
    tcrs = [get_tcr(i) for i in data.itertuples()] # get TCRs in the format (v_call, junction_aa, junction, j_call)
    nt_seq = [capture_nucseq(i) for i in tcrs] # extract the CDR3 nucleotide sequence
    data["junction"] = nt_seq # replace junction column
    data = data.dropna(subset=["junction"]) # drop TCRs with faulty rearrangements
    n_post = data.shape[0] # number of TCRs in data after parsing
    print(f"dropped {n_pre-n_post} TCRs with ambiguous junctions")
    return data

def check_formatting(data : pd.DataFrame, verbose=False):

    # Check if data contains all the necessary columns
    if verbose:
        print("Check if necessary columns are present")
    colnames = data.columns
    required = ["v_call", "j_call", "junction", "junction_aa"]
    for req in required:
        assert req in colnames, f"DataFrame missing column: {req}"
    
    # Check if V/J genes are formatted correctly
    if verbose:
        print("Check V and J gene formatting")
    functional = IMGT[IMGT['fct'].isin(['F','(F)','[F]'])]
    v_check = data.v_call.isin(functional.imgt_allele_name).eq(True).all()
    j_check = data.v_call.isin(functional.imgt_allele_name).eq(True).all()
    assert v_check, "Column 'v_call' formatted improperly\n \
        -- see https://www.imgt.org/IMGTrepertoire/index.php?section=LocusGenes&repertoire=genetable&species=human&group=TRBV (IMGT allele name)for the correct formatting."
    assert j_check, "Column 'j_call' formatted improperly\n \
        -- see https://www.imgt.org/IMGTrepertoire/index.php?section=LocusGenes&repertoire=genetable&species=human&group=TRBJ (IMGT allele name)for the correct formatting."

    # Check formatting of junction
    if verbose:
        print("Check if junction is correctly formatted")
    junction_nt_check = (data.junction.str.len() / data.junction_aa.str.len() == 3).eq(True).all()
    assert junction_nt_check, "CDR3 nucleotide sequence length does not match CDR3 amino acid length\n \
        -- use the 'raptcr.constants.parsing.format_junction' function to correctly parse the junction"
    junction_aa_check = data.junction_aa.apply(lambda x: _is_cdr3(x)).eq(True).all()
    assert junction_aa_check, "Column 'junction_aa' contains non amino-acid characters"

    if verbose:
        print("All checks completed -- DataFrame correctly formatted")


def parse_repertoire(
    file,
    cdr3aa_col="cdr3_b_aa",
    cdr3nt_col="cdr3_b_nucseq",
    vgene_col="v_b_gene",
    jgene_col="j_b_gene",
    sep="\t",
    compression=None,
    drop_cols=None
    ):

    col_remap = {
        cdr3nt_col:"junction",
        cdr3aa_col:"junction_aa",
        vgene_col:"v_call",
        jgene_col:"j_call",
        }
    
    df = pd.read_csv(file, sep=sep, compression=compression)  
    df = df.rename(columns=col_remap)

    # Remove ambiguous CDR3 amino acid sequences
    print("Remove ambiguous CDR3 amino acid sequences")
    df = df[df.junction_aa.apply(lambda cdr3: _is_cdr3(cdr3))]
    
    # Remove ORF and non-functional genes
    print("Parsing V/J genes")
    functional = IMGT[IMGT['fct']=='F']
    df = df[df.v_call.isin(functional.imgt_allele_name)]
    
    # Parse junction (CDR3 nucleotide sequence)
    print("Parsing CDR3 nucleotide sequence")
    df["junction"] = [capture_nucseq((l.v_call, l.junction_aa, l.junction, l.j_call)) for l in df.itertuples()]
    df = df.dropna(subset=["junction_aa","junction","v_call","j_call"])
    
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    
    return df

def parse_adaptive_repertoire(name, old=False, recover_unresolved=True):
    # Read file
    df = pd.read_csv(name, sep="\t", low_memory=False)
    # Remove out-of-frame sequences
    df = df[df.frame_type=="In"]
    # Desired columns
    if old:
        prefix = "cdr3_"
    else:
        prefix = ""
    cols = [
        'templates',
        f'{prefix}rearrangement', 
        f'{prefix}amino_acid',
        'v_family',
        'j_family',
        'v_gene',
        'v_allele',
        'j_gene',
        'j_allele'
        ]
    df = df[cols]
    # Gene parsing
    unresolved = df[df.v_gene=="unresolved"]
    df["v_imgt"] = df.v_gene.map(adaptive_to_imgt_human)
    df["j_imgt"] = df.j_gene.map(adaptive_to_imgt_human)
    df = df.dropna(subset=["v_imgt","j_imgt"])
    unresolved["v_imgt"] = unresolved.v_family.map(adaptive_vfam_mapping)
    unresolved["j_imgt"] = unresolved.j_gene.map(adaptive_to_imgt_human)
    if recover_unresolved:
        print(f"Recovered {unresolved.v_imgt.dropna().shape[0]} V genes")
        df = pd.concat([df,unresolved])
    df[["v_allele","j_allele"]] = df[["v_allele","j_allele"]].fillna(0)
    # # df["v_call"] = df.apply(lambda v: v.v_imgt.split("*")[0] + f"*0{int(v.v_allele)}" if v.v_allele!=0 else v, axis=1)
    # # df["j_call"] = df.apply(lambda j: j.j_imgt.split("*")[0] + f"*0{int(j.j_allele)}" if j.j_allele!=0 else j, axis=1)
    df = df.dropna(subset=["v_imgt"])
    print(df.shape)
    df["v_call"] = [df.v_imgt.iloc[n].split("*")[0]+f"*0{int(i)}" if int(i)!=0 else df.v_imgt.iloc[n] for n,i in enumerate(df.v_allele)]
    df["j_call"] = df.j_imgt
    functional = IMGT[IMGT['fct']=='F']
    df = df[df.v_call.isin(functional.imgt_allele_name)]
    df = df[['templates',f'{prefix}rearrangement',f'{prefix}amino_acid','v_call','j_call']]
    df = df[df[f"{prefix}amino_acid"].apply(lambda cdr3: _is_cdr3(cdr3))]
    df = df.dropna(subset=[f'{prefix}rearrangement',f'{prefix}amino_acid','v_call','j_call'])
    df = df.sort_values(by="templates", ascending=False)
    df = df.rename(columns={f"{prefix}rearrangement":"junction",f"{prefix}amino_acid":"junction_aa"})
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def parse_britanova_repertoires(file, count=2):

    df = pd.read_csv(file, sep='\t', compression='gzip')

    cols = {
        'cdr3nt':'junction', 
        'cdr3aa':'junction_aa', 
        'v':'v_call', 
        'j':'j_call',  
        'count':'clone_count', 
        'freq':'clone_fraction'
        }

    df = df[df['count']>=count]
    df = df[list(cols.keys())]
    df = df.rename(columns=cols)

    # Parse VJ into IMGT format
    df['v_call'] = df['v_call'].apply(lambda x: x.split('(')[0])
    df['v_call'] = df['v_call'].apply(lambda x: x.split('*')[0] + '*01')
    df['j_call'] = df['j_call'].apply(lambda x: x.split('(')[0])
    df['j_call'] = df['j_call'].apply(lambda x: x.split('*')[0] + '*01')

    # Remove ORF and pseudogenes
    functional = IMGT[IMGT['fct']=='F']
    df = df[df.v_call.isin(functional.imgt_allele_name)]
    # Remove ambiguous CDR3 sequences
    df = df[df['junction_aa'].apply(lambda x: _is_cdr3(x))]

    return df.reset_index(drop=True)

def parse_yfv_repertoires(file, output='regular', count=2):
    

    if file.split('.')[-1]=='gz':
        df = pd.read_csv(file, compression='gzip', sep='\t', low_memory=False)
    else:
        df = pd.read_csv(file, sep='\t', low_memory=False)

    # Columns of interest
    if 'AA. Seq. CDR3' in df.columns:
        cols = {
            'AA. Seq. CDR3':'junction_aa', 
            'N. Seq. CDR3':'junction', 
            'All V hits':'v_call', 
            'All J hits':'j_call', 
            'Clone ID':'clonotype_id', 
            'Clone count':'clone_count', 
            'Clone fraction':'clone_fraction'
            }
        # Filter out singlets
        df = df[df['Clone count']>=count]
    else:
        cols = {
            'CDR3.amino.acid.sequence':'junction_aa', 
            'CDR3.nucleotide.sequence':'junction', 
            'bestVGene':'v_call', 
            'bestJGene':'j_call'
            }

    df = df[list(cols.keys())]
    df = df.rename(columns=cols)

    # Parse VJ into IMGT format
    df['v_call'] = df['v_call'].apply(lambda x: x.split('(')[0])
    df['v_call'] = df['v_call'].apply(lambda x: x.split('*')[0] + '*01')
    df['j_call'] = df['j_call'].apply(lambda x: x.split('(')[0])
    df['j_call'] = df['j_call'].apply(lambda x: x.split('*')[0] + '*01')

    # Remove ORF and pseudogenes
    functional = IMGT[IMGT['fct']=='F']
    df = df[df.v_call.isin(functional.imgt_allele_name)]
    # Remove ambiguous CDR3 sequences
    df = df[df['junction_aa'].apply(lambda x: _is_cdr3(x))]

    if output == 'regular':
        return df.reset_index(drop=True)
    elif output == 'alternative':
        cols = {
            'junction_aa':'cdr3aa',
            'junction':'cdr3nt',
            'v_call':'v',
            'j_call':'j',
            'clone_count':'count'
            }
        df = df.rename(columns=cols)
        df = df[list(cols.values())]
        df['cdr3nt'] = df.cdr3nt.str.lower()
        return df.reset_index(drop=True)

def read_sne_file(path, correct_pvals=True, filter_significant=True):
    name = os.path.basename(path).split('_')[0]
    enriched = pd.read_csv(path, sep="\t")
    if correct_pvals:
        enriched["p_adj"] = fdrcorrection(enriched.pval)[1]
    if filter_significant:
        enriched = enriched[enriched.p_adj<.05]
    enriched["clone"] = enriched["v_call"]+"_"+enriched["junction_aa"]
    enriched["sample"] = name
    return enriched
