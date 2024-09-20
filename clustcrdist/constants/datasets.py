import pandas as pd
import random
from os.path import dirname, abspath, join

DIR = dirname(abspath(__file__))
vdjdb_location = join(DIR,'data/vdjdb/vdjdb.txt')

# def sample_cdr3s(s:int, background:str=FILENAME, aa_col:str='junction_aa') -> list:
#     '''
#     Randomly sample s number of cdr3s from the background distribution.
#     '''
#     n = sum(1 for line in open(FILENAME)) - 1 # number of records in file (excludes header)
#     skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
#     return pd.read_csv(background, sep='\t', skiprows=skip)[aa_col].to_list()

# def sample_tcrs(s:int, background:str=FILENAME, cols:list=['junction_aa','v_call','j_call']):
#     n = sum(1 for line in open(FILENAME)) - 1 # number of records in file (excludes header)
#     skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
#     return pd.read_csv(background, sep='\t', skiprows=skip)[cols]

# def load_yfv_responding():
#     responding = pd.read_csv("./analysis/background_testing/yfv/yfv_responding.txt", sep="\t")
#     responding = responding[["bestVGene","CDR3.amino.acid.sequence","donor"]].drop_duplicates()
#     responding.columns = ["v_gene","junction_aa","donor"]
#     responding["responding"] = True
#     return responding

def load_vdjdb(chain="B", organism="human", exclude_10x=True, concise=True):
    
    vdjdb = pd.read_csv(vdjdb_location, sep="\t")
    vdjdb.columns = [i.replace(".","_") for i in vdjdb.columns]

    ref_map = {
        "cdr3":"junction_aa",
        "v_segm":"v_call",
        "j_segm":"j_call",
    }

    vdjdb = vdjdb.rename(columns=ref_map)
    
    if chain.upper() == "B":
        vdjdb = vdjdb[vdjdb.gene=="TRB"]
    else:
        vdjdb = vdjdb[vdjdb.gene=="TRA"]
    
    if organism.lower() == "human":
        vdjdb = vdjdb[vdjdb.species=="HomoSapiens"]
    elif organism.lower() == "mouse":
        vdjdb = vdjdb[vdjdb.species=="MusMusculus"]
    else:
        pass

    if exclude_10x:
        tenx_id = vdjdb["reference_id"].value_counts().index[0]
        vdjdb = vdjdb[vdjdb["reference_id"]!=tenx_id]

    if concise:
        vdjdb = vdjdb[["junction_aa","v_call","j_call","antigen_epitope","antigen_species"]]
    
    vdjdb = vdjdb.drop_duplicates()
    
    return vdjdb.reset_index(drop=True)
