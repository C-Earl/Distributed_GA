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

  # Handles logic for generating new gene given state of the pool
  @abstractmethod
  def fetch_gene(self, **kwargs):
    pass

  # Take gene and write it to a file. Returns file name and written data
  def create_gene_file(self, gene: Union[np.array, dict]):
    # Generate gene & name
    gene_key = get_pool_key(gene)

    # Write gene to file
    gene_data = {'gene': gene, 'fitness': None, 'status': 'being tested'}
    write_gene(gene_data, gene_key, self.run_name)

    # Return gene/file info
    return gene_key

  # Remove gene from pool (in files)
  def delete_gene_file(self, gene_name: str):
    delete_gene(gene_name, self.run_name)

  # Create class vars with proper typing
  # Note: bash args always returned as strings
  def make_class_vars(self, **kwargs):
    for arg, arg_value in kwargs.items():
      setattr(self, arg, arg_value)


class Evolutionary_Algorithm_Base(Algorithm):
  def __init__(self, num_genes: int, **kwargs):
    self.num_genes = num_genes
    super().__init__(**kwargs)

  def fetch_gene(self, **kwargs):

    # Only use tested parents
    valid_parents = {gene_key: gene_data for gene_key, gene_data in self.pool.items()  # Filter untested genes
                     if (not gene_data['status'] == 'being tested')}

    # If pool is unitialized, add new gene (phase 1)
    if len(self.pool.items()) < self.num_genes:
      new_gene = self.initial_gene(**kwargs)
      gene_name = self.create_gene_file(new_gene)
      return gene_name, True

    # If more than half of the pool is untested, wait.
    elif len(valid_parents.items()) < (self.num_genes / 2):
      return None, False

    # Otherwise, create a new offspring
    else:

      # Simplify pool and create new gene (user defined)
      simplified_pool = {   # Send only gene and fitness (no status)
        gene_key: {'gene' : gene_data['gene'], 'fitness' : gene_data['fitness']}
        for gene_key, gene_data in valid_parents.items()
      }
      new_gene = self.create_new_gene(simplified_pool, **kwargs)
      gene_name = self.create_gene_file(new_gene)

      # Update pool in files/class (above ref to pool should contain changes)
      for gene_key, gene_data in simplified_pool.items():
        if gene_key not in valid_parents.keys():     # gene was added
          self.create_gene_file(gene_data)
          self.pool[gene_key] = {'gene': gene_data, 'fitness': None, 'status': 'being tested'}
      for gene_key, gene_data in valid_parents.items():
        if gene_key not in simplified_pool.keys():   # gene was removed
          self.delete_gene_file(gene_key)
          del self.pool[gene_key]

      return gene_name, True

  @abstractmethod
  # Create initial gene
  def initial_gene(self, **kwargs):
    pass

  @abstractmethod
  def create_new_gene(self, gene_pool: dict, **kwargs):
    pass


class Genetic_Algorithm(Evolutionary_Algorithm_Base):

  def __init__(self, gene_shape: tuple, mutation_rate: float, **kwargs):
    self.gene_shape = gene_shape
    self.mutation_rate = mutation_rate
    super().__init__(**kwargs)

  # Create new gene from current state of pool
  def create_new_gene(self, gene_pool: dict, **kwargs):

    # Initial pool manipulation (remove worst gene)
    gene_pool = self.pool_manipulation(gene_pool)

    # Select parents for reproduction
    p1, p2 = self.select_parents(gene_pool)

    # Generate offspring with crossover
    new_gene = self.crossover(p1, p2)

    # Random mutation
    new_gene = self.mutate(new_gene)

    return new_gene

  @abstractmethod
  # Manipulate pool before selection
  def pool_manipulation(self, gene_pool: dict):
    pass

  @abstractmethod
  # Select parents from pool
  def select_parents(self, gene_pool: dict):
    pass

  @abstractmethod
  # Crossover parents to create offspring
  def crossover(self, p1, p2):
    pass

  @abstractmethod
  # Mutate offspring
  def mutate(self, gene):
    pass
