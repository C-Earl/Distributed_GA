import copy
from os.path import join as file_path
import os
import hashlib
import numpy as np
from abc import abstractmethod
from typing import Union
from DGA.File_IO import load_gene_file
from DGA.Pool import Pool, Subset_Pool

POOL_DIR = "pool"
POOL_LOCK_NAME = "POOL_LOCK.lock"

# Return hash of bytes of obj x
def consistent_hasher(x):
  b = bytes(str(x), 'utf-8')
  return hashlib.sha256(b).hexdigest()  # Get the hexadecimal representation of the hash value

# Takes np and transforms into tuple (makes it hashable)
def hashable_nparray(gene: np.ndarray):
  if gene.ndim == 0:  # Scalar value
    return gene.item()
  else:
    return tuple(hashable_nparray(sub_arr) for sub_arr in gene)


def get_pool_key(gene: np.ndarray | dict):
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

  # Constructor for all algorithms
  # - Loads pool from files
  # - Undefined vars are set to -1 so they won't be saved to files (issues with inheritence & loading args)
  def __init__(self,
               num_genes: int = -1,
               gene_shape: Union[tuple, dict] = (-1,),
               mutation_rate: float = -1.0,
               iterations: int = -1,
               **kwargs):

    # Algorithm specific vars. Only turned to class vars if defined by user
    if num_genes != -1:
      self.num_genes = num_genes
    if gene_shape != (-1,):
      self.gene_shape = gene_shape
    if mutation_rate != -1.0:
      self.mutation_rate = mutation_rate
    if iterations != -1:
      self.iterations = iterations
    self.current_iter = kwargs.pop('current_iter', 0)     # Not an argument that needs to be set by user

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
    self.pool = Pool()        # Subset_Pool() will have all elements of Pool() meeting condition
    self.valid_parents = Subset_Pool(condition=lambda key, gene: gene['test_state'] == 'tested')
    self.pool.add_subset_pool(self.valid_parents)

    # Load gene pool
    self.pool.extend(self.load_gene_pool())

  def load_gene_pool(self):
    pool = {}
    for root, dirs, files in os.walk(self.pool_path):
      for file in files:
        file_name = file.split('.')[0]  # This will be unique hash of the gene
        gene = load_gene_file(self.run_name, file_name)
        pool[file_name] = gene
    return pool

  # Create class vars with proper typing
  def make_class_vars(self, **kwargs):
    for arg, arg_value in kwargs.items():
      setattr(self, arg, arg_value)

  @abstractmethod
  def fetch_gene(self, **kwargs) -> tuple:
    pass

  @abstractmethod
  def create_new_gene(self, **kwargs):
    pass

  @abstractmethod
  # Create initial gene to populate pool
  def initial_gene(self, **kwargs):
    pass

  @abstractmethod
  # Manipulate pool before selection
  def remove_weak(self):
    pass

  @abstractmethod
  # Select parents from pool
  def select_parents(self) -> tuple:
    pass

  @abstractmethod
  # Crossover parents to create offspring
  def crossover(self, p1, p2) -> dict:
    pass

  @abstractmethod
  # Mutate offspring
  def mutate(self, gene) -> dict:
    pass

  @abstractmethod
  # End condition for run
  def end_condition(self):
    pass


class Genetic_Algorithm(Genetic_Algorithm_Base):

  def __init__(self,
               num_genes: int = -1,
               gene_shape: Union[tuple, dict] = (-1,),
               mutation_rate: float = -1.0,
               iterations: int = -1,
               **kwargs):
    super().__init__(num_genes, gene_shape, mutation_rate, iterations, **kwargs)

  # Fetch a gene from the pool for testing. This function is a filter, checking that it is safe to create a new gene.
  # Handles pool-initialization logic, and ensures create_new_gene is only called when it's safe to do so (e.g. enough
  # genes in pool to make new gene).
  # - iterates current_iter
  # - Users should always call super function to ensure create_new_gene gets called
  def fetch_gene(self, **kwargs) -> tuple:
    self.current_iter += 1    # Increment iteration

    # If pool is unitialized, add new gene
    if len(self.pool.items()) < self.num_genes:
      new_gene = self.initial_gene(**kwargs)
      gene_name = get_pool_key(new_gene)
      self.pool[gene_name] = {'gene': new_gene,       # Add to pool obj
                              'fitness': None,
                              'test_state': 'being tested',
                              'iteration': self.current_iter}
      return gene_name, True

    # If there aren't at least 2 genes in pool, can't create new gene
    elif len(self.valid_parents.items()) < 2:
      self.current_iter -= 1    # No gene created, so don't increment (cancels out prior += 1)
      return None, False

    # Otherwise, create a new offspring
    else:
      new_gene = self.create_new_gene(**kwargs)
      gene_name = get_pool_key(new_gene)      # np.array alone cannot be used as key in dict
      while gene_name in self.pool.keys():    # Keep attempting until unique
        new_gene = self.create_new_gene(**kwargs)
        gene_name = get_pool_key(new_gene)

      # Remove worst gene(s) from the pool
      self.remove_weak()
      self.pool[gene_name] = {'gene': new_gene,       # Add to pool obj
                              'fitness': None,
                              'test_state': 'being tested',
                              'iteration': self.current_iter}
      return gene_name, True

  # Create new gene from current state of pool
  # - Called by fetch_gene to create new genes
  def create_new_gene(self, **kwargs):
    p1, p2 = self.select_parents()                # Select parents for reproduction
    new_gene = self.crossover(p1, p2)             # Generate offspring with crossover
    new_gene = self.mutate(new_gene)              # Random mutation
    return new_gene

  # Return randomized gene of shape gene_shape
  def initial_gene(self, **kwargs):
    if isinstance(self.gene_shape, tuple | list):
      return np.random.rand(*self.gene_shape)
    elif isinstance(self.gene_shape, dict):
      return {key: np.random.rand(*shape) for key, shape in self.gene_shape.items()}
    else:
      raise ValueError(f"gene_shape must be tuple, list or dict, not {self.gene_shape}")

  # Remove worst gene from pool
  def remove_weak(self):
    sorted_parents = sorted(self.valid_parents.items(),           # Only use tested genes
        key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)  # Sort by fitness
    worst_gene = sorted_parents[-1][0]
    del self.pool[worst_gene]  # Remove from pool

  # Select parents for reproduction using weighted probabilities based on fitness. Higher fitness -> higher probability of selection
  def select_parents(self) -> tuple:
    fitness_scores = [gene_data['fitness'] for _, gene_data in self.valid_parents.items()]  # Get fitness's (unordered)
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
    p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)	# Select 2 parents
    sorted_genes = sorted(self.valid_parents.items(), key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)
    return sorted_genes[p1_i][1]['gene'], sorted_genes[p2_i][1]['gene']

  # Crossover p1 and p2 genes
  def crossover(self, p1, p2, gene_shape=None) -> dict:
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
  def mutate(self, gene, gene_shape=None) -> dict:
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
  def end_condition(self) -> bool:
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


class Plateau_Genetic_Algorithm(Genetic_Algorithm):

  # - plateau_sensitivity: How steep the fitness curve must be to be considered a plateau
  #         Smaller values -> more sensitive to plateau
  # - plateau_sample_size: How many past fitness's to observe for plateau
  #         Smaller values -> less accurate detection of plateau
  # - iterations_per_epoch: Max number of genes to test before starting new epoch
  # - epochs: Max number of epochs to run before stopping
  # - Note: iterations is not used for logic in this algorithm (but still needed for constructor).
  #         replaced with iterations_per_epoch
  def __init__(self,
               num_genes: int,
               gene_shape: tuple | dict,
               mutation_rate: float,
               mutation_decay: float,
               plateau_sensitivity: float,
               plateau_sample_size: int,
               iterations_per_epoch: int,
               epochs: int, warmup: int,
               past_n_fitness: list = None,
               founders_pool: dict = None,
               **kwargs):
    super().__init__(num_genes, gene_shape, mutation_rate, **kwargs)
    self.mutation_decay = mutation_decay
    self.plateau_sensitivity = plateau_sensitivity
    self.plateau_sample_size = plateau_sample_size
    self.iterations_per_epoch = iterations_per_epoch
    self.epochs = epochs
    self.warmup = warmup
    self.past_n_fitness = past_n_fitness if past_n_fitness is not None else (np.ones(self.plateau_sample_size) * -np.inf)
    self.current_epoch = kwargs.pop('current_epoch', 0)   # Not an argument that needs to be set by user
    self.epoch_iter = kwargs.pop('epoch_iter', 0)
    self.original_mutation_rate = kwargs.pop('original_mutation_rate', mutation_rate)  # Save original rate for resetting on epochs
    self.founders_pool = founders_pool if founders_pool is not None else {}

  # Additional checks made to ensure new gene is safe to create
  # - Must use super().fetch_gene to ensure create_new_gene is called and current_iter is iterated
  def fetch_gene(self, **kwargs):
    self.current_iter += 1  # Total iteration
    self.epoch_iter += 1    # Iteration for current epoch
    org_decay = self.mutation_decay # Save original decay rate
    self.mutation_rate *= self.mutation_decay

    # If pool is unitialized
    # Initialize new gene, add to pool, and return
    if len(self.pool.items()) < self.num_genes:
      new_gene = self.initial_gene(**kwargs)
      gene_name = get_pool_key(new_gene)  # np.array alone cannot be used as key in dict
      while gene_name in self.pool.keys():  # Keep attempting until unique
        new_gene = self.initial_gene(**kwargs)
        gene_name = get_pool_key(new_gene)
      self.pool[gene_name] = {'gene': new_gene,       # Add to pool
                              'fitness': None,
                              'founder_proximity_penalty': self.founder_proximity_penalty(new_gene),
                              'test_state': 'being tested',
                              'iteration': self.current_iter}
      return gene_name, True

    # If there aren't at least 2 tested genes in pool, can't create new gene
    # Return no gene and unsuccessful flag (False)
    elif len(self.valid_parents.items()) < 2:
      self.current_iter -= 1
      self.epoch_iter -= 1
      self.mutation_decay = org_decay
      return None, False   # Any changes made during this iteration will be undone

    # Check if max iterations for an epoch
    # Start new epoch and initialize new gene
    elif self.epoch_iter >= self.iterations_per_epoch:
      self.start_new_epoch()
      new_gene = self.initial_gene()
      gene_name = get_pool_key(new_gene)
      self.pool[gene_name] = {'gene': new_gene,
                              'fitness': None,
                              'founder_proximity_penalty': self.founder_proximity_penalty(new_gene),
                              'test_state': 'being tested',
                              'iteration': self.current_iter}
      return gene_name, True

    ## Add most recent fitness's to list ##
    # Shift frame for past_n_fitness & add any newly tested genes
    self.past_n_fitness = np.roll(self.past_n_fitness, -1)
    for gene_key, gene in self.valid_parents.items():
      if gene['iteration'] in range(self.current_iter - self.plateau_sample_size, self.current_iter):
        ind = self.plateau_sample_size - (self.current_iter - gene['iteration'])
        self.past_n_fitness[ind] = gene['fitness']

    # Check for performance plateau (new epoch if plateau detected & past warmup phase)
    coefs = np.polyfit(np.arange(len(self.past_n_fitness)), self.past_n_fitness, 1)
    if coefs[0] < self.plateau_sensitivity and self.epoch_iter > self.warmup:  # If slope is small enough
      self.start_new_epoch()
      new_gene = self.initial_gene()
      gene_name = get_pool_key(new_gene)
      self.pool[gene_name] = {'gene': new_gene,  # Add to pool obj
                              'fitness': None,
                              'founder_proximity_penalty': self.founder_proximity_penalty(new_gene),
                              'test_state': 'being tested',
                              'iteration': self.current_iter}
      return gene_name, True

    # Otherwise, breed new offspring & return
    else:
      new_gene = self.create_new_gene(**kwargs)
      gene_name = get_pool_key(new_gene)    # np.array alone cannot be used as key in dict
      while gene_name in self.pool.keys():  # Keep attempting until unique
        new_gene = self.create_new_gene(**kwargs)
        gene_name = get_pool_key(new_gene)
      self.remove_weak()
      self.pool[gene_name] = {'gene': new_gene,       # Add to pool obj
                              'fitness': None,
                              'founder_proximity_penalty': self.founder_proximity_penalty(new_gene),
                              'test_state': 'being tested',
                              'iteration': self.current_iter}
      return gene_name, True

  # Create new gene from current state of pool
  # - Called by fetch_gene to create new genes
  def create_new_gene(self, **kwargs):
    p1, p2 = self.select_parents()  # Select parents for reproduction
    new_gene = self.crossover(p1, p2)  # Generate offspring with crossover
    new_gene = self.mutate(new_gene)  # Random mutation
    return new_gene

  # Remove worst gene from pool
  def remove_weak(self):
    sorted_parents = sorted(self.valid_parents.items(),  # Only use tested genes
                            key=lambda gene_kv: gene_kv[1]['fitness'] + gene_kv[1]['founder_proximity_penalty'], reverse=True)  # Sort by fitness
    worst_gene = sorted_parents[-1][0]
    del self.pool[worst_gene]  # Remove from pool

  # Select parents for reproduction using weighted probabilities based on fitness. Higher fitness -> higher probability of selection
  def select_parents(self) -> tuple:
    gene_list = list(self.valid_parents.items())
    fitness_scores = [gene_data['fitness'] + gene_data['founder_proximity_penalty'] for _, gene_data in gene_list]
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
    p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)  # Select 2 parents
    return gene_list[p1_i][1]['gene'], gene_list[p2_i][1]['gene']

  # Crossover p1 and p2 genes
  def crossover(self, p1, p2, gene_shape=None) -> dict:
    return super().crossover(p1, p2, gene_shape)

  # Mutate at random point in gene
  def mutate(self, gene, gene_shape=None) -> dict:
    return super().mutate(gene, gene_shape)

  # Begin a new epoch, and return the first gene of that epoch
  def start_new_epoch(self, **kwargs):
    self.current_epoch += 1
    self.mutation_rate = self.original_mutation_rate
    self.epoch_iter = 0
    self.past_n_fitness = (np.ones(self.plateau_sample_size) * -np.inf)

    # Move top scoring genes to founders pool
    sorted_gene_fitness = sorted(self.valid_parents.items(), key=lambda gene_kv: gene_kv[1]['fitness'] + gene_kv[1]['founder_proximity_penalty'], reverse=True)
    top_gene_key, top_gene_data = sorted_gene_fitness[0]
    while top_gene_key in self.founders_pool.keys():      # Ensure no duplicates
      sorted_gene_fitness = sorted_gene_fitness[1:]
      top_gene_key, top_gene_data = sorted_gene_fitness[0]
    self.founders_pool[top_gene_key] = top_gene_data

    # Re-initialize pool
    # Note: When other model-process's return, fetch_gene will handle filling the pool
    for gene_key, gene in list(self.valid_parents.items()):
      del self.pool[gene_key]

  # Special case; apply penalty based on proximity to founders
  def remove_weak(self):
    sorted_parents = sorted(    # Sort by fitness + penalty
        self.valid_parents.items(),
        key=lambda gene_kv:
        gene_kv[1]['fitness'] + gene_kv[1]['founder_proximity_penalty'],
        reverse=True
    )
    worst_gene = sorted_parents[-1][0]
    del self.pool[worst_gene]  # Remove from pool obj

  # Diversity based on cumulative Euclidean distance between gene and other genes in pool
  # def get_diversity(self, pool: dict):
  #   return sum([np.linalg.norm(gene - other_gene) for gene in pool.values() for other_gene in pool.values()])

  # Penalty for being close to founders
  # Return positive L2 distance between gene and all genes in founders pool
  def founder_proximity_penalty(self, gene):
    return sum([np.linalg.norm(gene - founder_gene['gene']) for founder_gene in self.founders_pool.values()])

  # Full override of end_condition. Only end when max epochs reached
  def end_condition(self):
    if self.current_epoch >= self.epochs:
      return True
