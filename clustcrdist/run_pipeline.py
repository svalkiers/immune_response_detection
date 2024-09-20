import argparse
import pandas as pd
import numpy as np

from os import path, mkdir, getcwd, listdir
from typing import Union
from clustcrdist.neighbors import neighbor_analysis

def parse_separator(filename):
    if filename.split(".")[-1] == "tsv":
        return "\t"
    elif filename.split(".")[-1] == "txt":
        return "\t"
    else:
        return ","

def analyze_file(
    filename,
    ratio=10,
    radius=12.5,
    chain="B",
    organism="human",
    suffix="",
    outdir=None,
    custom_background:Union[str,pd.DataFrame]=None,
    # custom_index=None,
    downsample=None
    ):

    """
    Perform neighborhood enrichment analysis for a single repertoire.

    Parameters:
    -----------
    filename    str
        The path to the input file.
    ratio   float, optional
        The ratio between background and foreground. Default is 10.
    radius  float, optional
        The radius for defining neighbors. Default is 12.5.
    suffix  str, optional
        A suffix to add to the output file name. Default is an empty string.
    outdir  str or None, optional
        The output directory path. If None, the current working directory is used. Default is None.
    custom_background   str or None, optional
        The path to a custom background file. 
        If provided, it will be used for computing the ratio and background data. Default is None.
    custom_index    str or None, optional
        The path to a custom background index file. 
        If provided, it will be loaded for neighbor enrichment analysis. Default is None.
    downsample  int or None, optional)
        The number of sequences to downsample from the input file. 
        If None, no downsampling is performed. Default is None.

    Returns:
    pvals   pandas.DataFrame
        A DataFrame containing the computed p-values.

    Note:
    - The input file should be in CSV format, and the separator will be automatically determined based on the file extension.
    - If downsampling is performed, a fixed number of sequences will be randomly sampled from the input data.
    - If a custom background file is provided, it will be used to calculate the ratio, which represents the ratio of background data to the input data.
    - The neighbor enrichment analysis is performed using the radius and the background data (if provided).
    - If a custom background index file is provided, it will be loaded to enhance the background index used for neighbor enrichment analysis.
    - The computed p-values are saved to a TSV file in the specified output directory (or the current working directory if not provided) with the input file name as the base name and the suffix added.

    Example usage:
    >>> analyze_file("data.csv", ratio=5, radius=12.5, suffix="enriched", outdir="output/", custom_background="background.csv")
    """
    df = pd.read_csv(filename, sep=parse_separator(filename))
    # Downsample to fixed number of sequences (optional)
    if downsample is not None:
        df = df.sample(downsample)
    # Load custom background
    if custom_background is not None:
        print("Using user-provided background")
        if isinstance(custom_background, str):
            background = pd.read_csv(custom_background, sep=parse_separator(custom_background))
        elif isinstance(custom_background, pd.DataFrame):
            background = custom_background
        ratio = background.shape[0] / df.shape[0]
    else:
        background = None
    
    res = neighbor_analysis(
        tcrs = df,
        chain = chain,
        organism = organism,
        radius = radius,
        background = background,
        depth = ratio
    )

    # enricher = NeighborEnrichment(
    #     repertoire=df,
    #     background=background
    #     )
    
    # if custom_index is not None:
    #     print("Loading user-provided background index")
    #     enricher.load_background_index_from_file(custom_index)
    #     ratio = enricher.bg_index.idx.ntotal / df.shape[0]

    # print("Computing neighbors in foreground...")
    # enricher.fixed_radius_neighbors(radius=radius)
    # pvals = enricher.compute_pvalues(ratio=ratio)

    base = path.basename(filename)
    file_out = base.split('.')[0] + f'_{suffix}.tsv'

    if outdir is None:
        outdir = getcwd()
    elif not path.isdir(outdir):
        mkdir(outdir)
    else:
        pass

    path_out = path.join(outdir,file_out)
    res.to_df().to_csv(path_out, sep="\t", index=False)

    return res.to_df()

def analyze_directory(
    indir,
    ratio=10,
    radius=12.5,
    chain='B',
    organism='human',
    suffix:str="",
    outdir:str=None,
    custom_background:str=None,
    custom_index:str=None,
    downsample:int=None
    ):

    """

    Parameters:
    -----------
    custom_background
        The path to a custom background file or directory. If a directory is provided, background files should have 
        the same name as the input repertoire they correspond to. In addition, they should contain a _background suffix.
        If provided, it will be used for computing the ratio and background data. Default is None.
    """
    assert path.isdir(indir), f"{indir} is not a directory. Please provide a path to the directory that contains the input files."

    files = listdir(indir)
    if custom_background is not None:
        if path.isdir(custom_background):
            backgrounds = {bg.split("_background")[0]:bg for bg in listdir(custom_background)}
            for f in files:
                fname = path.join(indir,f)
                prefix = f.split(".")[0]
                analyze_file(
                    filename=fname,
                    ratio=ratio,
                    radius=radius,
                    chain=chain,
                    organism=organism,
                    suffix=suffix,
                    outdir=outdir,
                    custom_background=backgrounds[prefix],
                    # custom_index=None,
                    downsample=None
                    )
                
        elif path.isfile(custom_background):
            background = custom_background
            for f in files:
                fname = path.join(indir,f)
                prefix = f.split(".")[0]
                analyze_file(
                    filename=fname,
                    ratio=ratio,
                    radius=radius,
                    chain=chain,
                    organism=organism,
                    suffix=suffix,
                    outdir=outdir,
                    custom_background=background,
                    # custom_index=None,
                    downsample=None
                    )
    else:
        for f in files:
            fname = path.join(indir,f)
            prefix = f.split(".")[0]
            print(fname)
            analyze_file(
                filename=fname,
                ratio=ratio,
                radius=radius,
                chain=chain,
                organism=organism,
                suffix=suffix,
                outdir=outdir,
                custom_background=custom_background,
                # custom_index=None,
                downsample=None
                )
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=None, help="Path to the file that contains the repertoire of interest. \
        When analyzing multiple files at once, please use the 'directory' argument.")
    parser.add_argument('-d', '--directory', type=str, default=None, help="Path to the directory that contains the repertoires of interest. \
        When analyzing a single file, please use the 'filename' argument.")
    parser.add_argument('-r', '--radius', default=12.5, help="The radius for defining neighbors. Default is 12.5.")
    parser.add_argument('-q', '--ratio', type=int, default=10, help="The ratio between background and foreground. \
        Only applicable when no custom background is provided. Default is 10.")
    parser.add_argument('-c', '--chain', type=str, default="B", help="TCR chain. AB for alphabeta. Default is B.")
    parser.add_argument('-s', '--species', type=str, default="human", help="Species. Default is human.")
    parser.add_argument('-x', '--suffix', type=str, default="", help="A suffix to add to the output file name.")
    parser.add_argument('-o', '--outdir', type=str, required=True, help="Path to directory where results will be saved. \
        If directory is non-existent, a new one will be created.")
    parser.add_argument('--custom_background', default=None, help="The path to a custom background file. ")
    parser.add_argument('--downsample', type=int, default=None, help="The number of sequences to downsample from the input file. \
        Default is None.")
    
    args = parser.parse_args()

    radius = float(args.radius)
    ratio = int(args.ratio)

    if args.filename is not None:
        analyze_file(
            filename=args.filename,
            ratio=ratio,
            radius=radius,
            chain=args.chain,
            organism=args.species,
            suffix=args.suffix,
            outdir=args.outdir,
            custom_background=args.custom_background,
            downsample=args.downsample
        )
    elif args.directory is not None:
        analyze_directory(
            indir=args.directory,
            ratio=ratio,
            radius=radius,
            chain=args.chain,
            organism=args.species,
            suffix=args.suffix,
            outdir=args.outdir,
            custom_background=args.custom_background,
            downsample=args.downsample
        )

if __name__ == "__main__":
    main()

# if args.filename.split(".")[1] == "tsv":
#     sep = "\t"
# elif args.filename.split(".")[1] == "txt":
#     sep = "\t"
# else:
#     sep = ","

# df = pd.read_csv(args.filename, sep=sep)
# if args.downsample is not None:
#     df = df.sample(args.downsample)

# r = float(args.radius)

# enricher = NeighborEnrichment(
#     repertoire=df, 
#     radius=r, 
#     background=args.custom_background
#     )

# if args.custom_index is not None:
#     enricher.load_background_index_from_file(args.custom_index)
# print("Computing neighbors in foreground...")
# enricher.compute_neighbors()
# pvals = enricher.compute_pvalues()

# if not path.isdir(args.outdir):
#     mkdir(args.outdir)

# base = path.basename(args.filename)
# file_out = base.split('.')[0] + f'_{args.suffix}.tsv'
# path_out = path.join(args.outdir,file_out)
# pvals.to_csv(path_out, sep="\t", index=False)