# Detection of antigen-driven convergent T-cell responses

[![PyPI](https://img.shields.io/pypi/v/clustcrdist.svg)](https://pypi.org/project/clustcrdist/)

The **clustcrdist** library allows for statistical quantification of TCR sequence similarity through enrichment analysis of sequence neighbor counts. **clustcrdist** builds on the concept of TCR neighborhoods introduced in [Mayer-Blackwell et al. (2021), eLife](https://elifesciences.org/articles/68605). The library makes use of efficient vectorization in order to compute neighbor distributions. In addition it uses a novel strategy for generating TCR repertoire backgrounds that match important properties of the input repertoire such as V/J gene frequency, CDR3 length and non-templated nucleotide insertion in the CDR3. 

## Installation

**clustcrdist** is available as a [pypi package](https://pypi.org/project/clustcrdist/). To install the package, simply run:

```sh
pip install clustcrdist
```

> ⚠️ **Note:** Make sure you have Fortran compiler like `gfortran`, which is necessary for running certain dependencies of the software.

The package makes use of multiple sequence alignment (MSA) for some of the downstream visualizations. If you wish to use those functionalities, make sure to install the MSA tool.

Linux (Ubuntu):

```bash
sudo apt-get install mafft
```

macOS:

```bash
brew install mafft
```

Windows:

https://mafft.cbrc.jp/alignment/software/windows_without_cygwin.html

## Use

### Command line interface

By far the easiest way to run the analysis is through the using of the command line interface.

```
usage: clustcrdist [-h] [-f FILENAME] [-d DIRECTORY] [-r RADIUS] [-q RATIO] [-c CHAIN] [-s SPECIES] [-x SUFFIX] -o OUTDIR [--custom_background CUSTOM_BACKGROUND] [--custom_index CUSTOM_INDEX] [--downsample DOWNSAMPLE]

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Path to the file that contains the repertoire of interest. When analyzing multiple files at once, please use the 'directory' argument.
  -d DIRECTORY, --directory DIRECTORY
                        Path to the directory that contains the repertoires of interest. When analyzing a single file, please use the 'filename' argument.
  -r RADIUS, --radius RADIUS
                        The radius for defining neighbors. Default is 12.5.
  -q RATIO, --ratio RATIO
                        The ratio between background and foreground. Only applicable when no custom background is provided. Default is 10.
  -c CHAIN, --chain CHAIN
                        TCR chain. AB for alphabeta. Default is B.
  -s SPECIES, --species SPECIES
                        Species. Default is human.
  -x SUFFIX, --suffix SUFFIX
                        A suffix to add to the output file name.
  -o OUTDIR, --outdir OUTDIR
                        Path to directory where results will be saved. If directory is non-existent, a new one will be created.
  --custom_background CUSTOM_BACKGROUND
                        The path to a custom background index file.
  --custom_index CUSTOM_INDEX
                        The path to a custom background file.
  --downsample DOWNSAMPLE
                        The number of sequences to downsample from the input file. Default is None.
```

**Example:**

```bash
clustcrdist --filename ./clustcrdist/data/test.tsv --chain AB --species human --radius 96 --ratio 10 --suffix result --outdir ./testresult/
```

**Note:**

When analyzing **multiple files**, the `-f` or `--filename` should remain **unspecified**. Instead  `-d` or `--directory` should be used.

### Accepted formats

The method accepts data in the [AIRR format](https://docs.airr-community.org/en/stable/datarep/rearrangements.html) or the paired TCRdist format. Data in the AIRR format should contain at least the following columns: *v_call*, *j_call*, *junction*, *junction_aa*. In case of using paired chain data, make sure the AIRR-formated data also includes the columns *cell_id* and *locus*. Data in the paired TCRdist format should contain at least the following columns: *va*, *ja*, *cdr3a*, *cdr3a_nucseq*, *vb*, *jb*, *cdr3b*, *cdr3b_nucseq*.

### Advanced use (python interface)

Alternatively, the python interface can be used, which allows for more flexibility and provides additional functionalities. 

#### Data formatting

The first important step is correctly formatting the TCR repertoire data. This implies that your data contains TCRs that satisfy the following specific criteria:

- Only canonical CDR3 sequences (N-terminal cysteine (**C**) and C-terminal phenylalanine (**F**), tryptophan (**W**) or cysteine (**C**)) that contain at least 4 and at most 30 amino acids. Non-amino acid characters are not allowed.
- V/J genes should be IMGT-formatted and must not include non-functional variants (ORFs/pseudogenes).
- The CDR3 nucleotide sequence should exactly match the CDR3 amino acid sequence.

To facilitate this formatting, you can make use of the `Repertoire` class. 

It is important to distinguish datasets in the single column or paired column format. Note that paired chain data may also be in the single column format, when TCRα and TCRβ chains are linked by the `cell_id`  column. Here are some examples of data in the **single column format**:

```
# EXAMPLE 1 (TRB only)
junction_aa	junction	v_call	j_call
CASSPQFTGSYEQYF	TGCGCCAGCAGCCCCCAGTTCACAGGCTCCTACGAGCAGTACTTC	TRBV4-3*01	TRBJ2-7*01
CASSSPIAGQSSYEQYF	TGTGCCAGCAGTTCCCCCATAGCGGGACAAAGCTCCTACGAGCAGT...	TRBV28*01	TRBJ2-7*01
CASSYGQNYNEQFF	TGCGCCAGCAGCTACGGACAGAACTACAATGAGCAGTTCTTC	TRBV5-1*01	TRBJ2-1*01
...

# EXAMPLE 2 (paired chain, but single column)
v_call	j_call	junction_aa	cell_id	locus
TRAV20*01	TRAJ3*01	CAVQAGWEASKIIF	AAGACCTAGTACACCT-1	TRA
TRAV19*01	TRAJ52*01	CALSEGAGGTSYGKLTF	ACGCCGAGTCTCTTAT-1	TRA
TRBV13*01	TRBJ2-2*01	CASSLQGAKSTGELFF	AAGACCTAGTACACCT-1	TRB
...
```

In contrast, paired column data contains separate columns for TCRα and TCRβ. Here is an example of what the **paired column format** looks like:

```
# EXAMPLE
cdr3a	cdr3a_nucseq	va	ja	cdr3b	cdr3b_nucseq	vb	jb
CVVKILTGGGNKLTF	TGTGTGGTGAAGATACTCACGGGAGGAGGAAACAAACTCACCTTT	TRAV12-1*01	TRAJ10*01	CASSPLADSSGSSYEQYF	TGTGCCAGCTCACCTCTCGCCGACAGCTCAGGGAGCTCCTACGAGCAGTACTTC	TRBV18*01	TRBJ2-7*01
CILADTGTASKLTF	TGCATCCTGGCCGATACCGGCACTGCCAGTAAACTCACCTTT	TRAV26-2*01	TRAJ44*01	CASKERGGLYEQYF	TGTGCCAGCAAGGAGCGGGGGGGCCTTTACGAGCAGTACTTC	TRBV11-2*01	TRBJ2-7*01
CALDMDGNTPLVF	TGTGCTCTAGACATGGACGGAAACACACCTCTTGTCTTT	TRAV6*01	TRAJ29*01	CASSPRQGAGANVLTF	TGTGCCAGCAGCCCCAGACAGGGAGCCGGGGCCAACGTCCTGACTTTC	TRBV5-4*01	TRBJ2-6*01
...
```

The code blocks below shows an example of how the data would be formatted.

```python
from clustcrdist.repertoire import Repertoire
from clustcrdist.datasets import load_unformatted_example

# EXAMPLE: single column format
data = load_unformatted_example()
print(data.columns)
```

```
Index(['count', 'freq', 'cdr3nt', 'cdr3aa', 'v', 'd', 'j', 'VEnd', 'DStart',
       'DEnd', 'JStart'],
      dtype='object')
```

We then select the relevant columns ('cdr3nt', 'cdr3aa', 'v', 'j') and provide them to the formatter. The formatter will remove any TCR with ORF genes, misannotated V genes or non-canonical CDR3 sequences. If no allele information is provided, *01 will be assumed. The CDR3 nucleotide sequence is trimmed so that it matches the corresponding amino acid sequence. Finally, the TCR columns (when using the single column format) will be converted to the AIRR format.
```python
data_formatted = formatter.filter_and_format_single(
    cdr3aa_col = 'cdr3aa', # default
    cdr3nt_col = 'cdr3nt', # default
    vgene_col = 'v', # default
    jgene_col = 'j' # default
)
```

```
Remove ambiguous CDR3 amino acid sequences
Parsing V/J genes
Removed 230 with non-functional or misannotated V genes.
Parsing CDR3 nucleotide sequence
dropped 0 TCRs with ambiguous junctions
```

#### Calculating neighbor distributions

The `find_neighbors` returns the number of sequence neighbor for each TCR within a fixed TCRdist threshold radius.

``` python
from clustcrdist.datasets import load_test
from clustcrdist.neighbors import find_neighbors

tcrs = load_test(column_type='paired') # test data -> change to your own data here
nbrs = find_neighbors(
    tcrs = tcrs,
    chain = 'AB',
    radius = 96,
)
```

#### Sequence neighbor enrichment

To interpret the neighbor counts, `neighbor_analysis` performs a neighbor enrichment analysis. It compares the emperical neighbor distribution in the sample with an estimate the expected neighbor counts in a synthetic background sample. Below, we run the analysis for a single paired αβ chain repertoire using a TCRdist radius of < 96.

```python
 from clustcrdist.neighbors import neighbor_analysis

tcrs = load_test(column_type='paired') # test data -> change to your own data here
result = neighbor_analysis(
    tcrs = tcrs,
    chain = 'AB', # paired chain (alpha-beta)
    organism = 'human',
    radius = 96 # TCRdist distance radius
)
```

##### Clustering

After running the analysis, access the data in the `SneTcrResult` object. To to perform clustering on the enrichment results, run `.get_clusters()` before extracting the results.

```python
result.get_clusters(
    r = 96.5,
    periphery = True # if set to True, this will include the 'periphery' (all neighbors) around each SNE
) # this will add a 'cluster' column to the results table
clustered_results = result.to_df() # extracts the results table as a pandas.DataFrame
```

```python
clustered_results.sort_values('evalue')
```

```
          pvalue        evalue  tcr_index  radius  num_nbrs  \
20  5.875723e-18  2.229249e-14        342      96        13   
63  9.334463e-18  3.541495e-14       1202      96        14   
94  3.204331e-16  1.215723e-12       2116      96        12   
21  3.406484e-16  1.292420e-12        348      96        11   
22  1.931435e-15  7.327863e-12        353      96         8   

    expected_num_nbrs   bg_nbrs           va         ja             cdr3a  \
20           0.273160  103637.0    TRAV27*01  TRAJ52*01  CAGESAGGTSYGKLTF   
63           0.376645  142899.0  TRAV12-2*01  TRAJ23*01      CAVLNQGGKLIF   
94           0.276310  104832.0  TRAV12-2*01  TRAJ23*01      CAFLGQGGKLIF   
21           0.195859   74309.0  TRAV12-1*01  TRAJ31*01        CVVNEARLMF   
22           0.054837   20805.0    TRAV34*01  TRAJ52*01  CGADHAGGTSYGKLTF   

                                        cdr3a_nucseq           vb          jb  \
20  tgtgcaggagagagcgctggtggtactagctatggaaagctgacattt    TRBV27*01  TRBJ2-1*01   
63              tgtgccgtgctcaaccagggaggaaagcttatcttc  TRBV11-2*01  TRBJ1-1*01   
94              tgtgccttccttgggcagggaggaaagcttatcttc  TRBV11-2*01  TRBJ1-1*01   
21                    tgtgtggtgaacgaggccagactcatgttt  TRBV11-2*01  TRBJ1-1*01   
22  tgtggagcagaccatgctggtggtactagctatggaaagctgacattt    TRBV27*01  TRBJ2-2*01   

              cdr3b                                   cdr3b_nucseq  cluster  
20  CASSLMAGGLNEQFF  tgtgccagcagtttaatggcggggggcctgaatgagcagttcttc        3  
63    CASSLGLNTEAFF        tgtgccagcagcttagggctgaacactgaagctttcttt        1  
94    CASSLGVNTEAFF        tgtgccagcagcttaggggtgaacactgaagctttcttt        1  
21    CASSYGVNTEAFF        tgtgccagcagttacggggtgaacactgaagctttcttt        1  
22  CASRLLAGGIGELFF  tgtgccagcagattactagcgggggggatcggggagctgtttttt        4 
```

The resulting dataframe will contain all the results from the analysis:

- `pvalue` : uncorrected neighbor enrichment p-value.
- `evalue` : corrected version of the p-value. This value is obtained by multiplying the p-value by the size of the input data.
- `tcr_index` : index of the TCR in the original dataframe.
- `num_nbrs` : observed number of neighbors under the radius.
- `expected_num_nbrs` : baseline neighbor rate based estimated from the background.
- `bg_nbrs` : total number of neighbors observed in the background.

##### Visualization

After clustering, the results can be visualized as a network:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=150, figsize=(4,4))
result.draw_neighborhoods(
    ax = ax, 
    node_size = 'num_nbrs', # Will use the duplicate_count column to set the node size
    annotate = True # Adds the cluster labels
)
```

![neighborhoods_network.png](./fig/neighborhoods_network.png)

In addition, each cluster can be individually inspected to gain further insight into the V/J gene usage and the CDR3 amino acid motif.

```python
fig = result.draw_cluster(
    cluster_id=1, 
    labels=False, 
    node_size='num_nbrs'
)
```

![neighborhoods_network.png](./fig/cluster_example.png)

#### Calculating the pairwise distance matrix using vectorized TCRdist 

The function `compute_sparse_distance_matrix` computes pairwise distances among a set of TCRs, retaining only distnaces within the specified radius 'r'. It returns  a sparse distance matrix where zero-distances are encoded as -1. Here, *r* will determine the maximum distance that is included. Increasing *r* will slow down the computing time. Note that when *r* is very large, available memory resources may be exceeeded. 

```python
from clustcrdist.distance import compute_sparse_distance_matrix

tcrs = load_test(column_type='paired') # test data -> change to your own data here
dm = compute_sparse_distance_matrix(
    tcrs = tcrs,
    chain = 'AB',
    organism = 'human',
    r = 96
)
```

#### Generating background data

The class `BackgroundModel` generates a set of background TCRs that match a range of characteristics in the provided data. These include matching **V** and **J** gene frequency, **CDR3 amino acid length** distribution, and the number of **n-inserted nucleotides** in the CDR3. The size of the background can be specified as a factor of the input data. By default, the model will generate a background dataset that is 10 times the size of the input data.

```python
from clustcrdist.background import BackgroundModel

tcrs = load_test(column_type='paired') # test data -> change to your own data here
bgmodel = BackgroundModel(
	repertoire = tcrs,
	factor = 5 # relative to the size of the input data
)

background = bgmodel.shuffle(chain='AB') # specify the chain here
```

#### *vecTCRdist* (TCRdist-based TCR encoding)

The TCRdist-based encoding vecTCRdist is a transformation of the TCRdist distance matrix, that enables accurate approximations of TCRdist distances in euclidean space. vecTCRdist captures information from CDR1, CDR2, CDR2.5, and CDR3. By default, the CDR3 region is trimmed at the 3rd and 2nd to final position. Following trimming, gaps are introduced at a fixed position to standardize the length of each sequence. Every amino acid (including the gap character) is mapped to its corresponding numerical vector. These amino acid vectors are concatenated in the same order as they appear in the sequence. To obtain an encoding for each amino acid, we use the rows of the 21 x 21 TCRdist distance matrix, where each row and column represents an amino acid (including one gap character). These will then be projected into a lower dimensional representation using MDS.

![neighborhoods_network.png](./fig/encoding.png)

Below, a basic example of how the encoder can be used:

```python
from clustcrdist.encoding import TCRDistEncoder

encoder = TCRDistEncoder(
    aa_dim = 8, # number of dimensions per amino acid
    organism = 'human',
    chain = 'AB'
)

vecs = encoder.fit_transform(tcrs) # Create the vector embeddings
```

The TCRDistEncoder also offers slightly more advanced parametrization to adjust the trimming of the CDR3 or to assign more relative importance to the V gene region of the TCR:

```python
encoder = TCRDistEncoder(
    aa_dim = 8, # number of dimensions per amino acid
    organism = 'human',
    chain = 'AB',
    n_trim = 3, # number of amino acids trimmed from the start
    c_trim = 2, # number of amino acids trimmed from the end
    cdr3_weight = 3, # weight assigned to the CDR3 
    v_weight = 1 # weight assigned to the CDR1, CDR2 and CDR2.5 (V gene)
)

vecs = encoder.fit_transform(tcrs) # Create the vector embeddings
```
