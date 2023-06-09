import os
from os.path import join as file_path
import pickle

POOL_DIR = "pool"

# Write gene to file
def write_gene(gene: dict, name: str, run_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, name) + ".pkl"
  with open(gene_path, 'wb') as gene_file:
    pickle.dump(gene, gene_file)


# Load gene from file
def load_gene(name: str, run_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, name) + ".pkl"
  with open(gene_path, 'rb') as gene_file:
    gene = pickle.load(gene_file)
  return gene


# Delete gene file
def delete_gene(name: str, run_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, name) + ".pkl"
  os.remove(gene_path)
  return True