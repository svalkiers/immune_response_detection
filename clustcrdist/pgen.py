from .constants.modules.olga import load_model as load_model
from .constants.modules.olga import generation_probability as pgen
from .constants.modules.olga import sequence_generation as seq_gen
from .constants.preprocessing import format_chain
from os import path
from multiprocessing import Pool, cpu_count

import pandas as pd

chainmap = {'A':'alpha','B':'beta'}
DIR = '/'.join(path.dirname(path.abspath(__file__)).split('/')[:-1]) + '/'

def load_models(chain='B'):

    chain = format_chain(chain)
            
    params_file_name = path.join(DIR,f'clustcrdist/constants/modules/olga/default_models/human_T_{chainmap[chain]}/model_params.txt')
    marginals_file_name = path.join(DIR,f'clustcrdist/constants/modules/olga/default_models/human_T_{chainmap[chain]}/model_marginals.txt')
    V_anchor_pos_file = path.join(DIR,f'clustcrdist/constants/modules/olga/default_models/human_T_{chainmap[chain]}/V_gene_CDR3_anchors.csv')
    J_anchor_pos_file = path.join(DIR,f'clustcrdist/constants/modules/olga/default_models/human_T_{chainmap[chain]}/J_gene_CDR3_anchors.csv')
    
    if chain == 'B':
        genomic_data = load_model.GenomicDataVDJ()
        generative_model = load_model.GenerativeModelVDJ()
    elif chain == 'A':
        genomic_data = load_model.GenomicDataVJ()
        generative_model = load_model.GenerativeModelVJ()

    genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
    generative_model.load_and_process_igor_model(marginals_file_name)

    return generative_model, genomic_data

def init_pgen_model(chain='B'):
    """
    Initialize the OLGA model for PGEN calculations.
    """
    generative_model, genomic_data = load_models(chain=chain)
    return pgen.GenerationProbabilityVDJ(generative_model, genomic_data)

def init_seqgen_model(chain='B'):
    generative_model, genomic_data = load_models(chain=chain)
    if chain == 'B':
        return seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
    else:
        return seq_gen.SequenceGenerationVJ(generative_model, genomic_data)

def compute_pgen_for_sequence(pgen_model, seq):
    """
    Compute the generation probability for a single sequence.
    """
    cdr3, v, j = seq
    return pgen_model.compute_aa_CDR3_pgen(cdr3, v, j)

def calc_pgen(sequences, chain='B', ncpus=1):
    """
    Calculate the average generation probability of a cluster.
    PGEN calculations are based on the OLGA module.
    """
    
    if ncpus > cpu_count():
        print('Number of CPUs requested is greater than available CPUs. Using all available CPUs.')
        ncpus = cpu_count()

    pgen_model = init_pgen_model(chain=chain)

    with Pool(ncpus) as pool:
        results = pool.starmap(compute_pgen_for_sequence, [(pgen_model, seq) for seq in sequences])
    
    return results

def generate_sequence(seq_gen_model):
    return seq_gen_model.gen_rnd_prod_CDR3()

def generate_sequences(n, chain='B'):
    """
    Generate random sequences based on the OLGA model.
    """
    
    seq_gen_model = init_seqgen_model(chain=chain)
    
    vmap = vgenes(chain=chain)
    jmap = jgenes(chain=chain)
    
    seqs = pd.DataFrame(
        [generate_sequence(seq_gen_model) for seq in range(n)]
        )
    seqs.columns = ['junction','junction_aa','v_call','j_call']
    seqs['v_call'] = seqs['v_call'].map(vmap)
    seqs['j_call'] = seqs['j_call'].map(jmap)
    seqs['locus'] = chain
    return seqs

def vgenes(chain='B'):
    '''
    Load V gene reference file that contains mapping information to extract V alleles from indices.
    '''
    _, genomic_data = load_models(chain=chain)
    return {i:v[0] for i,v in enumerate(genomic_data.genV)}

def jgenes(chain='B'):
    _, genomic_data = load_models(chain=chain)
    return {i:j[0] for i,j in enumerate(genomic_data.genJ)}
