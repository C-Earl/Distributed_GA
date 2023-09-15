from os.path import join as file_path
import os
import numpy as np
import hashlib
from abc import abstractmethod
from typing import Union
from DGA.pool_functions import load_gene, write_gene, delete_gene, read_run_status, write_run_status, create_run_status

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


## Base class for genetic algorithm ##
# This is mostly for containing functions that most users won't need (e.g file management)
# and defining abstract methods
class Genetic_Algorithm_Base:

  def __init__(self, num_genes: int, gene_shape: Union[tuple, dict], mutation_rate: float, iterations: int, **kwargs):

    # Algorithm specific vars
    self.num_genes = num_genes
    self.gene_shape = gene_shape
    self.mutation_rate = mutation_rate
    self.iterations = iterations
    self.current_iter = kwargs.pop('current_iter', 0)

    # Check if run from user script or server
    # Note: When users start a run, they must pass a GA object with num_genes, gene_shape, and mutation_rate
    # into the Server obj. This is for syntax simplicity, but we can't load any run files because they don't
    # exist yet. Just return instead
    if 'run_name' not in kwargs.keys():
      return

    # File management vars
    # TODO: Maybe don't need make_class_vars
    self.run_name = kwargs.pop('run_name', None)
    self.make_class_vars(**kwargs)  # Initialize other vars (from inheriting classes)
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = {}

    # Load gene pool
    for root, dirs, files in os.walk(self.pool_path):
      for file in files:
        file_name = file.split('.')[0]  # This will be unique hash of the gene
        gene = load_gene(file_name, self.run_name)
        self.pool[file_name] = gene

  # Take gene and write it to a file. Returns file name and written data
  def create_gene_file(self, gene: Union[np.array, dict]):
    # Generate gene & name
    gene_key = get_pool_key(gene)

    # Write gene to file
    gene_data = {'gene': gene, 'fitness': None, 'test_state': 'being tested'}
    write_gene(gene_data, gene_key, self.run_name)

    # Return gene/file info
    return gene_key

  # Remove gene from pool (in files)
  def delete_gene_file(self, gene_name: str):
    delete_gene(gene_name, self.run_name)

  # Create class vars with proper typing
  def make_class_vars(self, **kwargs):
    for arg, arg_value in kwargs.items():
      setattr(self, arg, arg_value)

  # Fetch gene from pool; determines what gene should be tested next
  def fetch_gene(self, **kwargs):

    # Load run status
    # run status used to maintain states of run (e.g. current epoch) that need to be synchronized across multiple processes
    self.current_iter += 1    # Increment iteration

    # Only use tested parents
    valid_parents = {gene_key: gene_data for gene_key, gene_data in self.pool.items()  # Filter untested genes
                     if (not gene_data['test_state'] == 'being tested')}

    # If pool is unitialized, add new gene (phase 1)
    if len(self.pool.items()) < self.num_genes:
      new_gene = self.initial_gene(**kwargs)
      gene_name = self.create_gene_file(new_gene)
      return gene_name, True

    # If more than half of the pool is untested, wait.
    # TODO: Maybe better policy for this?
    elif len(valid_parents.items()) < (self.num_genes / 2):
      self.current_iter -= 1    # No gene created, so don't increment (cancels out prior += 1)
      return None, False

    # Otherwise, create a new offspring
    else:

      # Simplify pool and create new gene (user defined)
      simplified_pool = {   # Send only gene and fitness (no gene status)
        gene_key: {'gene' : gene_data['gene'], 'fitness' : gene_data['fitness']}
        for gene_key, gene_data in valid_parents.items()
      }
      new_gene = self.create_new_gene(simplified_pool, **kwargs)
      gene_name = self.create_gene_file(new_gene)
      simplified_pool[gene_name] = new_gene

      # Update pool in files/class (above ref to pool should contain changes)
      for gene_key, gene_data in simplified_pool.items():
        if gene_key not in valid_parents.keys():     # gene was added
          self.create_gene_file(gene_data)
          self.pool[gene_key] = {'gene': gene_data, 'fitness': None, 'test_state': 'being tested'}
      for gene_key, gene_data in valid_parents.items():
        if gene_key not in simplified_pool.keys():   # gene was removed
          self.delete_gene_file(gene_key)
          del self.pool[gene_key]

      return gene_name, True

  # Create new gene from current state of pool
  def create_new_gene(self, gene_pool: dict, **kwargs):

    # Initial pool manipulation (remove worst gene)
    gene_pool = self.remove_weak(gene_pool)

    # Select parents for reproduction
    p1, p2 = self.select_parents(gene_pool)

    # Generate offspring with crossover
    new_gene = self.crossover(p1, p2)

    # Random mutation
    new_gene = self.mutate(new_gene)

    return new_gene

  @abstractmethod
  # Create initial gene to populate pool
  def initial_gene(self, **kwargs):
    pass

  @abstractmethod
  # Manipulate pool before selection
  def remove_weak(self, gene_pool: dict):
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

  @abstractmethod
  # End condition for run
  def end_condition(self):
    pass


class Genetic_Algorithm(Genetic_Algorithm_Base):

  # Return randomized gene of shape gene_shape
  def initial_gene(self, **kwargs):
    if isinstance(self.gene_shape, tuple | list):
      return np.random.rand(*self.gene_shape)
    elif isinstance(self.gene_shape, dict):
      return {key: np.random.rand(*shape) for key, shape in self.gene_shape.items()}
    else:
      raise ValueError(f"gene_shape must be tuple, list or dict, not {self.gene_shape}")

  # Remove worst gene from pool
  def remove_weak(self, gene_pool: dict):
    sorted_parents = sorted(gene_pool.items(),
                            key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)  # Sort by fitness
    worst_gene = sorted_parents[-1][0]
    del gene_pool[worst_gene]  # Remove from pool obj
    return gene_pool

  # Select parents for reproduction
  def select_parents(self, gene_pool: dict):
    fitness_scores = [gene_data['fitness'] for _, gene_data in gene_pool.items()]  # Get fitness's (unordered)
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
    p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)
    sorted_genes = sorted(gene_pool.items(), key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)
    return sorted_genes[p1_i][1]['gene'], sorted_genes[p2_i][1]['gene']

  # Crossover p1 and p2 genes
  def crossover(self, p1, p2, gene_shape=None):
    if gene_shape is None:
      gene_shape = self.gene_shape

    if isinstance(p1, dict):
      return {key: self.crossover(p1[key], p2[key], self.gene_shape[key]) for key in p1.keys()}
    elif isinstance(p1, np.ndarray):
      crossover_point = np.random.randint(0, np.prod(gene_shape))       # 0 to end of gene_shape
      # crossover_index = np.unravel_index(crossover_point, gene_shape)   # Unravel index converts 1D index to nD index
      new_gene = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
      return new_gene.reshape(gene_shape)

  # Mutate at random point in gene
  def mutate(self, gene, gene_shape=None):
    if gene_shape is None:
      gene_shape = self.gene_shape

    if isinstance(gene, dict):
      return {key: self.mutate(val, self.gene_shape[key]) for key, val in gene.items()}
    elif isinstance(gene, np.ndarray):
      if np.random.rand() < self.mutation_rate:
        mutation_point = np.random.randint(0, np.prod(gene_shape))    # 0 to end of gene_shape
        gene[np.unravel_index(mutation_point, gene_shape)] += np.random.uniform(-self.mutation_rate, +self.mutation_rate)
        # ^ Unravel index converts 1D index to nD index
      return gene

  # End run when max iterations reached
  def end_condition(self):
    if self.current_iter >= self.iterations:
      return True

  # Normalize values to positive range [0, +inf) (fitnesses)
  # Do nothing if already in range [0, +inf)
  def pos_normalize(self, values):
    min_v = min(values)
    if min_v < 0:
      return [i + abs(min_v) for i in values]
    else:
      return values


# TODO: Get rid of super() calls? maybe confusing?
class Complex_Genetic_Algorithm(Genetic_Algorithm):

  # - plateau_sensitivity: How steep the fitness curve must be to be considered a plateau
  #         Smaller values -> more sensitive to plateau
  # - plateau_sensitivity: How many past fitness's to observe for plateau
  #         Smaller values -> less accurate detection of plateau
  # - max_iterations: Max number of genes to test before starting new epoch
  # - max_epochs: Max number of epochs to run before stopping
  def __init__(self, num_genes: int, gene_shape: tuple | dict, mutation_rate: float,
               plateau_sensitivity: float, plateau_sample_size: int, iterations_per_epoch: int,
               epochs: int, past_n_fitness: list = None, **kwargs):
    super().__init__(num_genes, gene_shape, mutation_rate, iterations=iterations_per_epoch, **kwargs)
    self.founders_pool = {}
    self.plateau_sensitivity = plateau_sensitivity
    self.plateau_sample_size = plateau_sample_size
    self.iterations_per_epoch = iterations_per_epoch
    self.epochs = epochs
    self.past_n_fitness = past_n_fitness if past_n_fitness is not None else []
    self.current_epoch = kwargs.pop('current_epoch', 0)   # Note: current_iter defined in base class

  def fetch_gene(self, **kwargs):

    # Check if max iterations for an epoch
    if self.current_iter >= self.iterations_per_epoch:
      self.start_new_epoch()

    # Check for performance plateau
    past_n_fitness = self.past_n_fitness
    if len(past_n_fitness) >= self.plateau_sample_size:   # If enough samples
      coefs = np.polyfit(np.arange(len(past_n_fitness)), past_n_fitness, 1)  # Get linear regression coefficients
      if coefs[0] < self.plateau_sensitivity:  # If slope is small enough
        self.start_new_epoch()

    # Check if max epochs
    if self.epochs >= self.max_epochs:
      return None, False      # TODO: Figure out how to do terminate signal

    
    self.iterations += 1
    super().fetch_gene(**kwargs)

  def start_new_epoch(self):
    self.current_epoch += 1
    self.past_n_fitness = []
    self.current_iter = 0

    # Move top scoring genes to founders pool
    top_gene_key, top_gene_data = sorted(self.pool.items(), key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)[0]
    self.founders_pool[top_gene_key] = top_gene_data

    # Re-initialize pool
    self.pool = {}
    for i in range(self.num_genes):
      new_gene = self.initial_gene()
      # TODO: This should be simplified
      self.pool[self.create_gene_file(new_gene)] = {'gene': new_gene, 'fitness': None, 'test_state': 'being tested'}

  # Special case; apply penalty based on proximity to founders
  def remove_weak(self, gene_pool: dict):
    sorted_parents = sorted(gene_pool.items(),
        key=lambda gene_kv: gene_kv[1]['fitness'] + self.founder_proximity_penalty(gene_kv[1]['fitness']), reverse=True)  # Sort by fitness + penalty
    worst_gene = sorted_parents[-1][0]
    del gene_pool[worst_gene]  # Remove from pool obj

  # Diversity based on cumulative Euclidean distance between gene and other genes in pool
  # TODO: User defined distance func
  def get_diversity(self, pool: dict):
    return sum([np.linalg.norm(gene - other_gene) for gene in pool.values() for other_gene in pool.values()])

  # Penalty for being close to founders
  def founder_proximity_penalty(self, gene):
    return sum([np.linalg.norm(gene - founder_gene) for founder_gene in self.founders_pool.values()])
