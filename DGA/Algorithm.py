from os.path import join as file_path
import os
import numpy as np
import hashlib
from abc import abstractmethod
from typing import Union
from DGA.pool_functions import load_gene, write_gene, delete_gene
from ast import literal_eval

POOL_DIR = "pool"
POOL_LOCK_NAME = "POOL_LOCK.lock"


# Return hash of bytes of obj x
def consistent_hasher(x):
  b = bytes(str(x), 'utf-8')
  return hashlib.sha256(b).hexdigest()  # Get the hexadecimal representation of the hash value


# Takes np and transforms into tuple (makes it hashable)
def hashable_nparray(gene: np.array):
  if gene.ndim == 0:  # Scalar value
    return gene.item()
  else:
    return tuple(hashable_nparray(sub_arr) for sub_arr in gene)


def get_pool_key(gene: Union[np.array, dict]):
  # np arrays not hashable, convert to tuple
  if isinstance(gene, np.ndarray):
    b = bytes(gene)
    return hashlib.sha256(b).hexdigest()
  elif isinstance(gene, dict):
    return consistent_hasher(gene)



# Assumed that pool is locked for duration of objects existence
class Algorithm():

  def __init__(self, run_name: str, **kwargs):
    self.run_name = run_name
    self.make_class_vars(**kwargs)
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = {}

    # Load gene pool
    for root, dirs, files in os.walk(self.pool_path):
      for file in files:
        file_name = file.split('.')[0]    # This will be unique hash of the gene
        gene = load_gene(file_name, self.run_name)
        self.pool[file_name] = gene

  # Behavior: Will add new genes until self.num_genes genes are present. After, new genes
  # created will replace gene with lowest fitness
  @abstractmethod
  def fetch_gene(self, **kwargs):
    pass

  # Take gene and write it to a file. Returns file name and written data
  def create_gene(self, gene: Union[np.array, dict]):
    # Generate gene & name
    gene_name = get_pool_key(gene)

    # Write gene to file
    gene_info = {'gene': gene, 'fitness': None, 'status': 'being tested'}
    write_gene(gene_info, gene_name, self.run_name)

    # Return gene/file info
    return gene_name

  # Remove gene from pool (in files)
  def delete_gene(self, gene_name: str):
    delete_gene(gene_name, self.run_name)

  # Create class vars with proper typing
  # Note: bash args always returned as strings
  def make_class_vars(self, **kwargs):
    for arg, arg_value in kwargs.items():
      setattr(self, arg, arg_value)


class Genetic_Algorithm(Algorithm):
  def fetch_gene(self, **kwargs):

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
      sorted_parents = sorted(valid_parents.items(), key=lambda gene_kv: gene_kv[1]['fitness'],
                              reverse=True)  # Sort by fitness
      worst_gene = sorted_parents[-1][0]
      self.delete_gene(worst_gene)  # Remove from file dir
      del self.pool[worst_gene]  # Remove from pool obj
      del valid_parents[worst_gene]  # Remove from pool obj

      # Select parents for reproduction
      fitness_scores = [gene_data['fitness'] for _, gene_data in valid_parents.items()]  # Get fitness's (unordered)
      normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
      probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
      p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)
      p1_gene, p2_gene = sorted_parents[p1_i][1]['gene'], sorted_parents[p2_i][1]['gene']

      # Generate offspring with crossover
      crossover_point = np.random.randint(0, self.gene_shape[0])
      new_gene = np.concatenate((p1_gene[:crossover_point], p2_gene[crossover_point:]))

      # Random mutation
      if np.random.rand() < 0.5:
        mutation_point = np.random.randint(0, self.gene_shape[0])
        new_gene[mutation_point] += np.random.uniform(-self.mutation_rate, +self.mutation_rate)

      # new_gene = np.random.rand(10)
      gene_name = self.create_gene(new_gene)
      return gene_name, True

  # Normalize values to positive range [0, +inf) (fitnesses)
  # Do nothing if already in range [0, +inf)
  def pos_normalize(self, values):
    min_v = min(values)
    if min_v < 0:
      return [i + abs(min_v) for i in values]
    else:
      return values