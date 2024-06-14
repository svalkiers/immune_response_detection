import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--subject', required = False, type = str, help = "Path to file containing SUBJECT TCRsto be searched against")
parser.add_argument('--query', required = False, type = str, help = "Path to file containing  QUERY TCRs, can be left out for subject vs. subject search")
parser.add_argument('--compile_subject_from', required = False, type = str, help = "Directory path for a set of files to be compiled into a subject repertoire") 
parser.add_argument('--save_subject_file', required = False, type = str)
parser.add_argument('--save_index_file', required = False, type = str)
parser.add_argument('--sep', default = "\t", type = str, help = "delim on TCR repertoires input files")
parser.add_argument('--endswith', default = "\t", type = str, help = ".csv")
parser.add_argument('--startswith', default = "\t", type = str, help = ".csv")
parser.add_argument('--cdr3aa_column', default='amino_acid')
parser.add_argument('--radius', default = 8.5, help = "TCRdist CDR3 unweighted max distance to store")
parser.add_argument('--outfile', required = True, type= str, help = "full path to output scipy.sparse csr_matrix")
parser.add_argument('--aa_dim', default=16)
parser.add_argument('--full_tcr', default=False)
parser.add_argument('--min_templates', default=0, type = int)
parser.add_argument('--templates_column', default= 'templates' , type = str, help = "use to limit attention to template counts above some threshold")
parser.add_argument('--max_files', default= 3000, help= "set lower for testing if you don't want to compile everything", type = int)
args = parser.parse_args()





"""



python km_vectorize_TRBV_tcrdist.py \
    --compile_subject_from /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/V01 \
    --outfile /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_npz/test_V01_V01.npz \
    --save_index_file /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_indices/V01.faiss.flat.index \
    --save_subject_file /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_indices/V01.faiss.csv \
    --min_templates 0  --max_files 3000

python km_vectorize_TRBV_tcrdist.py \
    --compile_subject_from /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/V16 \
    --outfile /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_npz/test_V16_V16.npz \
    --save_index_file /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_indices/V16.faiss.flat.index \
    --save_subject_file /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_indices/V16.faiss.csv \
    --min_templates 0  --max_files 3000

    python km_vectorize_TRBV_tcrdist.py \
    --compile_subject_from /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/V21 \
    --outfile /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_npz/test_V21_V12.npz \
    --save_index_file /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_indices/V21.faiss.flat.index \
    --save_subject_file /fh/scratch/delete90/gilbert_p/t1d/stratified_trbv/faiss_indices/V21.faiss.csv \
    --min_templates 0  --max_files 3000


"""

print(f"###\tLoading <raptcr> and other dependencies")
import os
import pandas as pd
import numpy as np
from faiss import write_index, read_index, IndexFlatL2
from progress.bar import Bar
import timeit
from scipy.sparse import vstack, save_npz
from raptcr.hashing import TCRDistEncoder
from raptcr.hashing import Cdr3Hasher, TCRDistEncoder
from raptcr.indexing import IvfIndex, FlatIndex
from raptcr.export import range_search_to_csr_matrix, query_function
from raptcr.export import index_neighbors_manual


start_time_total = timeit.default_timer()

def compile_fixed_TRBV(
  compdir, 
  max_files = args.max_files, 
  endswith = '.csv', 
  startswith = 'V',
  min_aa_len    = 6,
  max_aa_len    = 22,
  min_templates = args.min_templates, 
  cdr3aa_col    = args.cdr3aa_column, 
  templates_col = args.templates_column):
  """Helper function to build a database from all the files in a directory. 
  For instance this would point to a folder containing all TRBV01 TCRs
  from each person. Optionally allows generation of subject file if 
  it has not already been created"""
  fs = sorted([f for f in os.listdir(compdir) if f.endswith(endswith) and f.startswith(startswith)])
  print(fs)
  dfs = list()
  cnt = 0
  with Bar('Compiling TRBV FIXED Files...', max=min(len(fs),max_files)) as bar:
      for f in fs:
          if cnt < max_files:
              cnt = cnt + 1
              df = pd.read_csv(os.path.join(compdir, f))
              dfs.append(df)
              bar.next()
  dfall = pd.concat(dfs).reset_index(drop = True)
  ix1 = dfall[cdr3aa_col].str.len() > min_aa_len
  ix2 = dfall[cdr3aa_col].str.len() < max_aa_len
  ix3 = ~dfall[cdr3aa_col].str.contains('[^A-Z]')
  ix4 = dfall[templates_col] > min_templates
  #print(ix1.mean(), ix2.mean(), ix3.mean())
  ix = ix1 & ix2 & ix3 & ix4
  dfall = dfall[ix].reset_index(drop = True)
  return(dfall)

h = TCRDistEncoder( aa_dim = args.aa_dim , full_tcr = args.full_tcr)

if args.subject is None:
  print(f"### Compile from subject TCR set from: {args.compile_subject_from}")
  subject = compile_fixed_TRBV(args.compile_subject_from, startswith = args.startswith)
  # Check size, if more than 2 million focus on templates > 1, increase template size until 2 million or less.
  subject_n_rows = subject.shape[0]
  while(subject_n_rows > 4E6):
    print(f"Increasing min templates because {subject_n_rows} subject has too many rows")
    args.min_templates = args.min_templates + 1
    subject = subject[subject[args.templates_column] >args.min_templates].reset_index(drop = True)
    subject_n_rows = subject.shape[0]

  if args.save_subject_file is not None:
    print(f"### Writing: {args.save_subject_file} with {subject_n_rows} rows")
    subject.to_csv(args.save_subject_file, index = False)
  S = h.fit_transform(X=subject[args.cdr3aa_column].to_list())
  S = S.astype(np.float32)
  S_n_rows, S_n_cols = S.shape
else:
  S = pd.read_csv(args.subject,  sep = args.sep)
  S = h.fit_transform(X=subject[args.cdr3aa_column].to_list())
  S = S.astype(np.float32)
  S_n_rows, S_n_cols = S.shape

index = IndexFlatL2(S.shape[1])
print(f"###\t\tBuilding: {index} with {S_n_rows} sequences")
index.add(S)
if args.save_index_file is not None:
  write_index(index, args.save_index_file)

if args.query is None:
  print(f"###\tRecyling the subject data as the query for all vs. all")
  Q = S
  Q_n_rows, Q_n_cols = Q.shape
else:
  query = pd.read_csv(args.query,  sep = args.sep)
  Q = h.fit_transform(X=query[args.cdr3aa_column].to_list())
  Q = Q.astype(np.float32)
  Q_n_rows, Q_n_cols = Q.shape


if Q_n_rows > 10000:
    max_rows_per_part = 10000
    Qs = list()
    print(Q_n_rows, max_rows_per_part)
    # Break the large matrix into smaller parts, to limit memory consumption
    for start_row in range(0, Q_n_rows, max_rows_per_part):
        end_row = min(start_row + max_rows_per_part, Q_n_rows)
        part = Q[start_row:end_row]
        Qs.append(part)
else:
    Qs = [Q]

with Bar('Processing...', max=len(Qs)) as bar:
  csr_mat_list = list()
  for i,q in enumerate(Qs):
    lims, D, I = index.range_search(q, thresh=args.radius)
    csr_mat = range_search_to_csr_matrix(lims, D, I, (q.shape[0], S_n_rows)) # give sparse matrix correct dimensions
    csr_mat_list.append(csr_mat)
    bar.next()  

csr_mat_all = vstack(csr_mat_list)
## CONSIDER CHANGING TYPE TO UNSIGNED SMALL INT FOR MEMORY
print(f"###\t Writing scipy sparse matrix to {args.outfile}")
save_npz(args.outfile, csr_mat_all, compressed=True)
elapsed = timeit.default_timer() - start_time_total
print(f"###\t Total Time {round(elapsed, 3)} seconds")




