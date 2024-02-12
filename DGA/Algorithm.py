import copy
from os.path import join as file_path
import os
import hashlib
import numpy as np
from abc import abstractmethod
from typing import Union
from DGA.File_IO import load_params_file as load_param_file, load_history, load_params_file_async, load_history_async, \
  load_params_file, load_pool, load_pool_async
from DGA.Pool import Pool, Subset_Pool
from DGA.Gene import Gene, Genome, Parameters

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
               num_params: int = -1,  # Number of Parameters in pool
               iterations: int = -1,  # Number of iterations to run
               genome: Genome = None,  # Genome to use for creating new Parameters
               **kwargs) -> None:

    # Algorithm specific vars. Only turned to class vars if defined by user
    if num_params != -1:
      self.num_params = num_params
    if iterations != -1:
      self.iterations = iterations
    if genome is not None:
      self.genome = genome
    self.current_iter = kwargs.pop('current_iter', 0)  # Not an argument that needs to be set by user

    # Check if run from user script or server
    # Note: When users start a run, they must pass a GA object with num_params & iterations
    # into the Server obj. This is for syntax simplicity, but we can't load any run files because they don't
    # exist yet. Just return instead
    if 'run_name' not in kwargs.keys():
      return

    # Other keyword args
    history = kwargs.get('history', False)  # Get previous tested params if doing parallel run
    additional_log_vars = kwargs.get('log_vars', [])  # Get additional names of vars for logging

    # File management vars
    self.run_name = kwargs.pop('run_name')
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = Pool()  # Subset_Pool() will have all elements of Pool() meeting condition
    self.valid_parents = Subset_Pool(condition=self.tested_condition)
    self.pool.add_subset_pool(self.valid_parents)

    # Set history if specified
    if history:
      self.history = {}
    else:
      self.history = None

    # Initialize logging vars
    self.log_vars = ['timestamp', 'fitness', 'iteration']
    self.log_vars.extend(additional_log_vars)

  # Test condition for valid_parents subpool
  def tested_condition(self, key, params):
    return params.tested()

  # Load pool from files
  # Inputs: async_ (bool) - If True, load files assuming multiple processes are writing to pool files
  #                         If False, load files assuming only 1 process is writing to pool files
  # Outputs: None
  def load_pool(self, async_: bool):
    if async_:
      return load_pool_async(self.run_name)
    else:
      return load_pool(self.run_name)

  # Load history from files
  # Inputs: async_ (bool) - If True, load files assuming multiple processes are writing to pool files
  #                         If False, load files assuming only 1 process is writing to pool files
  # Outputs: None
  def load_history(self, async_: bool):
    if async_:
      return load_history_async(self.run_name)
    else:
      return load_history(self.run_name)

  @abstractmethod
  # Fetch a new Parameters from pool for testing.
  def fetch_params(self, **kwargs) -> tuple:
    pass

  @abstractmethod
  # Breed new offspring with parents selected from pool
  def breed(self, **kwargs):
    pass

  @abstractmethod
  # Manipulate pool before selection
  def trim_pool(self):
    pass

  @abstractmethod
  # Select parents from pool
  def select_parents(self) -> tuple:
    pass

  @abstractmethod
  # End condition for run
  def end_condition(self):
    pass

  # Create initial Parameters to populate pool
  # Inputs: current iteration, user-specific keyword args
  # Outputs: new Parameters
  def spawn(self, iteration: int) -> Parameters:
    return self.genome.initialize(iteration)

  # Crossover parents to create offspring (according to user provided Genome)
  # Inputs: list of Parameters (parents), current iteration
  # Outputs: new Parameters (offspring)
  def crossover(self, parents: list[Parameters], iteration: int) -> Parameters:
    return self.genome.crossover(parents, iteration)

  # Mutate Parameters (according to user provided Genome)
  # Inputs: Parameters
  # Outputs: Parameters (same object, mutated)
  def mutate(self, params: Parameters) -> Parameters:
    params = self.genome.mutate(params)
    return params


class Genetic_Algorithm(Genetic_Algorithm_Base):

  def __init__(self,
               num_params: int,
               iterations: int,
               genome: Genome,
               num_parents: int = 2,
               **kwargs) -> None:
    super().__init__(num_params, iterations, genome, **kwargs)
    self.num_parents = num_parents  # TODO: Implement this

  # Fetch a new Parameters from pool for testing.
  # Inputs: None
  # Outputs: tuple of (params_name, success_flag)
  def fetch_params(self, **kwargs) -> tuple:
    self.current_iter += 1  # Increment iteration

    # If pool is uninitialized, initialize new Parameters
    if len(self.pool.items()) < self.num_params:
      new_params = self.spawn(self.current_iter)
      params_name = get_pool_key(new_params)
      self.pool[params_name] = new_params
      return params_name, True

    # If there aren't at least 2 genes in pool, can't create new gene
    elif len(self.valid_parents.items()) < 2:
      self.current_iter -= 1  # No gene created, so don't increment (cancels out prior += 1)
      return None, False

    # Otherwise, create a new offspring
    else:
      new_params = self.breed(self.current_iter)
      params_name = get_pool_key(new_params)  # np.array alone cannot be used as key in dict
      while params_name in self.pool.keys():  # Keep attempting until unique
        new_params = self.breed(self.current_iter)
        params_name = get_pool_key(new_params)

      # Remove worst Parameters from the pool
      self.trim_pool()

      # Add new Parameters to pool & return
      self.pool[params_name] = new_params
      return params_name, True

  # Breed new offspring with parents selected from pool
  # Inputs: current iteration
  # Outputs: new Parameters (new offspring)
  def breed(self, iteration: int) -> Parameters:
    parents = self.select_parents()
    offspring = self.crossover(parents, iteration)
    offspring = self.mutate(offspring)
    return offspring

  # Create initial Parameters to populate pool
  # Inputs: current iteration, user-specific keyword args
  # Outputs: new Parameters
  def spawn(self, iteration: int, **kwargs) -> Parameters:
    return self.genome.initialize(iteration, **kwargs)

  # Removes Parameters with lowest fitness from pool
  # Inputs: None
  # Outputs: None
  def trim_pool(self) -> None:
    if len(self.valid_parents) > self.num_params:
      num_to_remove = len(self.valid_parents) - self.num_params
      sorted_params = self.sort_params(self.valid_parents)
      for i in range(num_to_remove):
        param_name = sorted_params[-(i+1)][0]
        del self.pool[param_name]

  # Select parents (for breeding) from pool based on fitness
  # Inputs: None
  # Outputs: list of Parameters (self.num_parents long)
  def select_parents(self) -> list[Parameters]:
    params_list = list(self.valid_parents.values())
    fitness_scores = [params.fitness for params in params_list]
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]

    # Ensure no probabilities are 0
    max_ind = np.argmax(probabilities)
    for i, p in enumerate(probabilities):
      if p == 0:
        probabilities[i] += 1e-5
        probabilities[max_ind] -= 1e-5

    # Select parents based on probabilities
    parent_inds = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities,
                                    size=self.num_parents)
    return [params_list[i] for i in parent_inds]

  # Crossover parents to create offspring (according to user provided Genome)
  # Inputs: list of Parameters (parents), current iteration
  # Outputs: new Parameters (offspring)
  def crossover(self, parents: list[Parameters], iteration: int) -> Parameters:
    return self.genome.crossover(parents, iteration)

  # Mutate Parameters (according to user provided Genome)
  # Inputs: Parameters
  # Outputs: Parameters (same object, mutated)
  def mutate(self, params: Parameters) -> Parameters:
    params = self.genome.mutate(params)
    return params

  # End condition for run. Ends after self.iterations parameters have been tested
  # Inputs: None
  # Outputs: bool (True if run should end)
  def end_condition(self) -> bool:
    return self.current_iter >= self.iterations

  # Normalize values to positive range [0, +inf) (fitnesses)
  # Do nothing if already in range [0, +inf)
  def pos_normalize(self, values):
    min_v = min(values)
    if min_v < 0:
      return [i + abs(min_v) for i in values]
    else:
      return values

  # Sort Parameters by fitness
  def sort_params(self, params_list: dict[str, Parameters]) -> list[tuple[str, Parameters]]:
    sorted_params = sorted(params_list.items(), key=lambda x: x[1].fitness, reverse=True)
    return sorted_params

