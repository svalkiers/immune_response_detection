import matplotlib.pyplot as plt
import logomaker
import pandas as pd
import numpy as np
import networkx as nx
import os

from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MafftCommandline

def multiple_sequence_alignment(cdr3s):

    sequences = [SeqRecord(Seq(i),id=f'{n}') for n, i in enumerate(cdr3s)]

    # Run MAFFT for alignment
    with open("unaligned_sequences.fasta", "w") as output_handle:
        for seq in sequences:
            output_handle.write(f">{seq.id}\n{seq.seq}\n")
    mafft_cline = MafftCommandline(input="unaligned_sequences.fasta")
    stdout, stderr = mafft_cline()

    # Write aligned sequences to a new file
    with open("aligned_sequences.fasta", "w") as output_handle:
        output_handle.write(stdout)

    # Read the aligned sequences
    alignment = AlignIO.read("aligned_sequences.fasta", "fasta")
    os.remove("unaligned_sequences.fasta")
    os.remove("aligned_sequences.fasta")

    return alignment

def position_frequency_matrix(alignment):

    alignment_len = alignment.get_alignment_length()

    # Build position frequency matrix
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY-')
    positions = range(alignment_len)
    pfm = pd.DataFrame(0, index=positions, columns=amino_acids)
    for record in alignment:
        for i, aa in enumerate(str(record.seq)):
            if aa in amino_acids:
                pfm.at[i, aa] += 1
            else:
                pfm.at[i, '-'] += 1  # Treat as gaps or handle accordingly
    pfm = pfm.div(pfm.sum(axis=1), axis=0)
    pfm = pfm.fillna(0)

    return pfm

def cdr3_logo(cluster, ax, n_trim=0, c_trim=0):

    if c_trim > 0:
        seqs = cluster.junction_aa.apply(lambda x: x[n_trim:-c_trim])
    else:
        seqs = cluster.junction_aa.apply(lambda x: x[n_trim:])

    alignment = multiple_sequence_alignment(seqs)
    pos_df = position_frequency_matrix(alignment)
    
    # seqlen = len(seqs.iloc[0])
    # aas = []
    # for i in seqs:
    #     aas += i
    # aas = sorted(list(set(aas)))

    # pos_matrix = np.zeros((seqlen,len(aas)))
    # aa_dict = {i:[0]*seqlen for i in aas}
    # for pos in range(seqlen):
    #     aacount = seqs.str[pos].value_counts()
    #     for i in aacount.index:
    #         aa_dict[i][pos] = aacount[i]
    # pos_df = pd.DataFrame(aa_dict)

    # create Logo object
    logo = logomaker.Logo(
        pos_df,
        ax=ax,
        font_name='Arial',
        color_scheme='dmslogo_funcgroup'
        )

    return pos_df

def plot_cluster(cluster, r, chain='B', organism='human', ax=None):

    from .neighbors import compute_sparse_distance_matrix

    matrix = compute_sparse_distance_matrix(cluster, chain=chain, organism=organism, d=r+.5, m=16)

    non_zero_values = matrix.data
    row_indices, col_indices = matrix.nonzero()

    mask = (non_zero_values >= 0)
    filtered_row_indices = row_indices[mask]
    filtered_col_indices = col_indices[mask]
    ids = np.column_stack((filtered_row_indices, filtered_col_indices))

    nodes = cluster.reset_index(drop=True).index.values
    edges = [(nodes[i[0]], nodes[i[1]]) for i in ids] 

    G = nx.Graph()
    G.add_nodes_from(list(nodes))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)

    coordinates = np.array(list(pos.values()))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    c = -np.log10(cluster.evalue)

    if ax is None:
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        cbar = plt.scatter(x,y,linewidths=0.5,edgecolor='black',alpha=1,c=c,cmap='viridis')
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        cbar = ax.scatter(x,y,linewidths=0.5,edgecolor='black',alpha=1,c=c,cmap='viridis')

