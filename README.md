# Detection of antigen-driven convergent T-cell responses

## Description

...

## Installation

To run the code in this repository, use the following instructions:

1. Clone the repository to your local machine:

   ```sh
   git clone https://github.com/svalkiers/immune_response_detection.git
   ```

2. Navigate to the repository.

3. Create an environment that supports all the dependencies:

   ```sh
   conda env create --file environment.yml
   ```

4. Activate the environment.

   ```sh
   conda activate immune_response_detection
   ```

## Use

### Command line interface

By far the easiest way to run the analysis is through the using of the command line interface, provided through the  `run_pipeline.py` script.

```
usage: run_pipeline.py [-h] [-f FILENAME] [-d DIRECTORY] [-r RADIUS] [-q RATIO] [-c CHAIN] [-s SPECIES] [-x SUFFIX] -o OUTDIR [--custom_background CUSTOM_BACKGROUND] [--downsample DOWNSAMPLE]

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
  -s SUFFIX, --suffix SUFFIX
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
python3 run_pipeline.py --filename ./data/test.tsv --chain AB --organism human --radius 96 --ratio 10 --suffix result --outdir ./testresult/
```

**Note:**

When analyzing **multiple files**, the `-f` or `--filename` should remain **unspecified**. Instead  `-d` or `--directory` should be used.

### Advanced use (python interface)

Alternatively, the python interface can be used, which allows for more flexibility and provides additional functionalities.  Before trying to run the code, make sure your working directory is correctly configured:

```
import os
os.chdir(".../github/immune_response_detection/") # Change me
```

#### Calculating neighbor distributions

The `find_neighbors` function provides a simple method for calculating the sequence neighbor distribution in your sample at a fixed TCRdist threshold *r*.

``` python
from snetcr.datasets import load_test

tcrs = snetcr.load_test() # test data -> change to your own data here
nbrs = find_neighbors(
    tcrs = tcrs,
    chain = 'AB',
    radius = 96,
)
```

#### Sequence neighbor enrichment

To add some interpretation to the neighbor counts, you can perform a neighbor enrichment analysis. This will compare the neighbor distribution in the sample with a synthetic background sample to estimate the expected neighbor counts. The code block below shows the most basic example where we run the analysis for a single paired &alpha;&beta; chain repertoire using a TCRdist radius of < 96.

 ```python
 from snetcr.neighbors import neighbor_analysis
 
 result = neighbor_analysis(
     tcrs = tcrs,
     chain = 'AB', # paired chain (alpha-beta)
     organism = 'human',
     radius = 96 # TCRdist distance radius
 )
 ```

After running the analysis, you can access the data in the `SneTcrResult` object. If you want to perform clustering on the enrichment results, you should run `.get_clusters()` before extracting the results.

```python
res.get_clusters() # this will add a 'cluster' column to the results table
clustered_results = res.to_df() # extracts the results table as a pandas.DataFrame
```

#### Calculating the pairwise distance matrix using vectorized TCRdist 

If you want to calculate the pairwise distances among a set of TCRs, you can simply run the example provided in this code block. This will produce a sparse distance matrix where zero-distances are encoded as -1. Here, *r* will determine the maximum distance that is included. Increasing *r* will slow down the computing time. Note that when *r* is very large, this may result in memory issues.

```python
from snetcr.distance import compute_sparse_distance_matrix

dm = compute_sparse_distance_matrix(
    tcrs = tcrs,
    chain = 'AB',
    organism = 'human',
    r = 96
)
```

#### Generating background data

The following functionality allows you to generate a set of background TCRs that match a range of characteristics in the provided data. These include matching **V** and **J** gene frequency, **CDR3 amino acid length** distribution, and the number of **n-inserted nucleotides** in the CDR3. The size of the background can be specified as a factor of the input data. By default, the model will generate a background dataset that is 10x the size of the input data.

```python
from snetcr.background import BackgroundModel

bgmodel = BackgroundModel(
	repertoire = tcrs,
	factor = 5 # relative to the size of the input data
)

background = bgmodel.shuffle(chain='AB') # specify the chain here
```

#### *vecTCRdist* (TCRdist-based TCR encoding)

The TCRdist-based encoding *vecTCRdist* is a transformation of the TCRdist distance matrix, that enables accurate approximations of TCRdist distances in euclidean space. *vecTCRdist* captures information from CDR1, CDR2, CDR2.5, and CDR3.

```python
from snetcr.encoding import TCRDistEncoder

encoder = TCRDistEncoder(
    aa_dim = 8, # number of dimensions per amino acid
    organism = 'human'
    chain = 'AB'
)
```

