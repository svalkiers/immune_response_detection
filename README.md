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
   conda env create --file environment.yaml
   ```

4. Activate the environment.

   ```sh
   conda activate immune_response_detecion
   ```

## Use

### Command line interface

By far the easiest way to run the analysis is through the using of the command line interface, provided through the  `run_pipeline.py` script.

```
usage: run_pipeline.py [-h] [-f FILENAME] [-d DIRECTORY] [-r RADIUS] [-q RATIO] [-s SUFFIX] -o OUTDIR [--custom_background CUSTOM_BACKGROUND] [--custom_index CUSTOM_INDEX]
                       [--downsample DOWNSAMPLE]

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

```
python3 run_pipeline.py --filename ./data/example_repertoire.tsv --radius 12.5 --ratio 10 --suffix result --outdir /path_to_folder/output/
```

**Note:**

When analyzing **multiple files**, the `-f` or `--filename` should remain **unspecified**. Instead  `-d` or `--directory` should be used.

### Advanced use (python interface)

Alternatively, the python interface can be used, which allows for more flexibility.  Before trying to run the code, make sure your working directory is correctly configured:

```
import os
os.chdir("/path_to_repository/") # Change me
```

The code block below shows the most basic example where we run the analysis for a single repertoire using a TCRdist radius of < 24.5.

 ```python
 import pandas as pd
 from raptcr.neighbors import NeighborEnrichment
 
 foreground = pd.read_csv("./test_df.csv")
 enricher = NeighborEnrichment(repertoire=foreground)
 enricher.fixed_radius_neighbors(radius=24.5) # Determine neighbors in foreground
 result = enricher.compute_pvalues()
 ```

#### Using a custom background

```python
import pandas as pd
from raptcr.neighbors import NeighborEnrichment

background = pd.read_csv("./custom_background.csv")
foreground = pd.read_csv("./test_df.csv")

enricher = NeighborEnrichment(repertoire=foreground, background=background)
enricher.fixed_radius_neighbors(radius=24.5) # Determine neighbors in foreground
result = enricher.compute_pvalues()
```

#### Beyond the fixed radius

For a given TCR *T*, determine the number of neighbors *σ* at different radii *d*, and select the smallest radius for each sequence *argmin(d)* such that *σ(d,T) > t*. Here, *t* is a certain threshold for the minimum number of required neighbors (e.g. 5).

```python
import pandas as pd
from raptcr.neighbors import NeighborEnrichment

background = pd.read_csv("./custom_background.csv")
foreground = pd.read_csv("./test_df.csv")

enricher = NeighborEnrichment(repertoire=foreground, background=background)
radii = [12.5,18.5,24.5,30.5,36.5]
enricher.flexible_radius_neighbors(radii=radii, t=5)
result = enricher.compute_pvalues()
```
