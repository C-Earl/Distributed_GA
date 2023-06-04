import time
import ast
from os import mkdir, rmdir
from os.path import join as file_path, isdir
from filelock import FileLock
import os
import argparse
import pickle
import numpy as np
import sys
import subprocess
import hashlib

POOL_DIR = "pool"
LOCK_DIR = "locks"


# Write gene to file
def write_gene(gene: dict, name: str, run_name: str):
  with FileLock(file_path(run_name, LOCK_DIR, name) + ".lock"):
    pool_path = file_path(run_name, POOL_DIR)
    gene_path = file_path(pool_path, name) + ".pkl"
    with open(gene_path, 'wb') as gene_file:
      pickle.dump(gene, gene_file)

  # TODO: TEST IF THIS IS NECESSARY ON ALL SYSTEMS
  # Expected behavior: Above 'with' statements should automatically
  # delete lock file, but isn't during testing. Manual delete here
  lock_path = file_path(run_name, LOCK_DIR, name) + ".lock"
  os.remove(lock_path)


# Load gene from file
def load_gene(name: str, run_name: str):
  with FileLock(file_path(run_name, LOCK_DIR, name) + ".lock"):
    pool_path = file_path(run_name, POOL_DIR)
    gene_path = file_path(pool_path, name) + ".pkl"
    with open(gene_path, 'rb') as gene_file:
      gene = pickle.load(gene_file)

  # TODO: TEST IF THIS IS NECESSARY ON ALL SYSTEMS
  # Expected behavior: Above 'with' statements should automatically
  # delete lock file, but isn't during testing. Manual delete here
  lock_path = file_path(run_name, LOCK_DIR, name) + ".lock"
  os.remove(lock_path)

  return gene


# Delete gene file
def delete_gene(name: str, run_name: str):
  with FileLock(file_path(run_name, LOCK_DIR, name) + ".lock"):
    pool_path = file_path(run_name, POOL_DIR)
    gene_path = file_path(pool_path, name) + ".pkl"
    os.remove(gene_path)

  # TODO: TEST IF THIS IS NECESSARY ON ALL SYSTEMS
  # Expected behavior: Above 'with' statements should automatically
  # delete lock file, but isn't during testing. Manual delete here
  lock_path = file_path(run_name, LOCK_DIR, name) + ".lock"
  os.remove(lock_path)

  return True


def consistent_hasher(x):
  b = bytes(str(x), 'utf-8')
  return hashlib.sha256(b).hexdigest()  # Get the hexadecimal representation of the hash value


# Takes np and transforms into tuple (makes it hashable)
def hashable_nparray(gene: np.array):
  if gene.ndim == 0:  # Scalar value
    return gene.item()
  else:
    return tuple(hashable_nparray(sub_arr) for sub_arr in gene)

# TODO: note to self, this should only ever be called once per gene
def get_pool_key(gene: np.array):
  # hashable_gene = hashable_nparray(gene)    # np arrays not hashable, convert to tuple
  b = bytes(gene)
  return hashlib.sha256(b).hexdigest()


class Algorithm():
  def __init__(self, run_name: str, gene_shape: tuple, mutation_rate: float, num_genes: int = 10, **kwargs):
    self.run_name = run_name
    self.gene_shape = gene_shape
    self.mutation_rate = mutation_rate
    self.num_genes = num_genes
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = {}

    # Load gene pool
    # for root, dirs, files in os.walk(self.pool_path):
    #   for file in files:
    #     gene = load_gene(file.split('.')[0], run_name)
    #     gene_key = get_pool_key(gene['gene'])
    #     self.pool[gene_key] = gene

  def create_gene(self):
    # Generate gene & name
    gene = np.random.rand(10)
    gene_name = get_pool_key(gene)

    # Write gene to file
    write_gene(gene, gene_name, self.run_name)

    # Return gene/file info
    return gene_name, {'gene': gene, 'fitness': None, 'status': 'being tested'}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:  # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())
  call_type = all_args.pop('call_type')

  if call_type == "init":
    for i in range(10):
      p = subprocess.Popen(["python3", "Algorithm.py", "--call_type=run",])

  else:
    # Init alg (loads gene pool)
    run_name = "test_dir"
    gene_shape = 10
    mutation_rate = 0.2
    alg = Algorithm(run_name, gene_shape, mutation_rate)

    # Generate gene with alg
    gene_name, new_gene = alg.create_gene()
    # print(new_gene, gene_name)

    # Read gene we just wrote
    gene = load_gene(gene_name, run_name)
    # print(gene)

    # Delete gene
    res = delete_gene(gene_name, run_name)
    # print(res)
