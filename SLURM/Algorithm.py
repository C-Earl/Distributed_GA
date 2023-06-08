import time
import ast
from os import mkdir, rmdir
from os.path import join as file_path, isdir
import os
import argparse
import pickle
import numpy as np
import sys
import subprocess
import hashlib

POOL_DIR = "pool"
POOL_LOCK_NAME = "POOL_LOCK.lock"

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
  # np arrays not hashable, convert to tuple
  b = bytes(gene)
  return hashlib.sha256(b).hexdigest()

# Assumed that pool is locked for duration of objects existence
class Algorithm():
  def __init__(self, run_name: str, gene_shape: tuple, mutation_rate: float, num_genes: int = 10, **kwargs):
    self.run_name = run_name
    self.gene_shape = gene_shape
    self.mutation_rate = mutation_rate
    self.num_genes = num_genes
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = {}

    # Load gene pool
    for root, dirs, files in os.walk(self.pool_path):
      for file in files:
        file_name = file.split('.')[0]    # This will be unique hash of the gene
        gene = load_gene(file_name, run_name)
        self.pool[file_name] = gene

  # Behavior: Will add new genes until self.num_genes genes are present. After, new genes
  # created will replace gene with lowest fitness
  def fetch_gene(self):

    # Only use tested parents
    valid_parents = {gene_key: gene_data for gene_key, gene_data in self.pool.items()  # Filter untested genes
                     if (not gene_data['status'] == 'being tested')}

    # If pool is unitialized, add new gene (phase 1)
    if len(self.pool.items()) < self.num_genes:
      new_gene = np.random.rand(10)
      gene_name = self.create_gene(new_gene)
      return gene_name, True

    # If more than half of the pool is untested, wait.
    elif len(valid_parents.items()) < (self.num_genes / 2):
      return None, False

    # Otherwise, drop lowest fitness and create new gene (phase 2)
    else:

      # Drop lowest fitness
      sorted_parents = sorted(valid_parents.items(), key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)  # Sort by fitness
      worst_gene = sorted_parents[-1][0]
      delete_gene(worst_gene, self.run_name)    # Remove from file dir
      del self.pool[worst_gene]                 # Remove from pool obj
      del valid_parents[worst_gene]             # Remove from pool obj

      # Select parents for reproduction
      fitness_scores = [gene_data['fitness'] for _, gene_data in valid_parents.items()]  # Get fitness's (unordered)
      normed_fitness = self.pos_normalize(fitness_scores)       # Shift fitness's to [0, +inf)
      probabilities = normed_fitness / np.sum(normed_fitness)   # Normalize to [0, 1]
      p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)
      p1_gene, p2_gene = sorted_parents[p1_i][1]['gene'], sorted_parents[p2_i][1]['gene']

      # Generate offspring with crossover
      crossover_point = np.random.randint(0, self.gene_shape)
      new_gene = np.concatenate((p1_gene[:crossover_point], p2_gene[crossover_point:]))

      # Random mutation
      if np.random.rand() < 0.5:
        mutation_point = np.random.randint(0, self.gene_shape)
        new_gene[mutation_point] += np.random.uniform(-self.mutation_rate, +self.mutation_rate)

      # new_gene = np.random.rand(10)
      gene_name = self.create_gene(new_gene)
      return gene_name, True

  # Take gene and write it to a file. Returns file name and written data
  def create_gene(self, gene: np.array):
    # Generate gene & name
    gene_name = get_pool_key(gene)

    # Write gene to file
    gene_info = {'gene': gene, 'fitness': None, 'status': 'being tested'}
    write_gene(gene_info, gene_name, self.run_name)

    # Return gene/file info
    return gene_name

  # Normalize values to positive range [0, +inf) (fitnesses)
  # Do nothing if already in range [0, +inf)
  def pos_normalize(self, values):
    min_v = min(values)
    if min_v < 0:
      return [i + abs(min_v) for i in values]
    else:
      return values


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:  # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())
  call_type = all_args.pop('call_type')

  RUN_NAME = "test_dir"
  GENE_SHAPE = 10
  MUTATION_RATE = 0.2
  NUM_GENES = 10

  if call_type == "init":
    alg = Algorithm(RUN_NAME, GENE_SHAPE, MUTATION_RATE, num_genes=NUM_GENES)
    init_genes = []
    for i in range(NUM_GENES):     # Generate initial 10 genes
      init_genes.append(alg.fetch_gene())

    for g_name, _ in init_genes:    # Call 1 client for each gene
      p = subprocess.Popen(["python3", "Algorithm.py", "--call_type=run", f"--gene_name={g_name}"])

  else:
    g_name = all_args['gene_name']
    raise Exception(g_name)
