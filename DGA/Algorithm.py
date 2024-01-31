import copy
from os.path import join as file_path
import os
import hashlib
import numpy as np
from abc import abstractmethod
from typing import Union
from DGA.File_IO import load_params_file as load_param_file, load_history
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
    # self.agent_id = kwargs.get('agent_id', None)  # Get agent id if doing parallel run
    history = kwargs.get('history', False)  # Get previous tested params if doing parallel run
    additional_log_vars = kwargs.get('log_vars', [])  # Get additional names of vars for logging

    # File management vars
    self.run_name = kwargs.pop('run_name')
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = Pool()  # Subset_Pool() will have all elements of Pool() meeting condition
    self.valid_parents = Subset_Pool(condition=self.tested_condition)
    self.pool.add_subset_pool(self.valid_parents)

    # Load pool
    self.pool.extend(self.load_pool())

    # Load history
    if history:
      self.history = self.load_history()
    else:
      self.history = None

    # Initialize logging vars
    self.log_vars = ['timestamp', 'fitness', 'iteration']
    self.log_vars.extend(additional_log_vars)

  def tested_condition(self, key, params):
    return params.tested()

  def load_pool(self):
    pool = {}
    for root, dirs, files in os.walk(self.pool_path):
      for file in files:
        file_name = file.split('.')[0]  # This will be unique hash of the param
        params = load_param_file(self.run_name, file_name)
        pool[file_name] = params
    return pool

  def load_history(self):
    return load_history(self.run_name)

  @abstractmethod
  def fetch_params(self, **kwargs) -> tuple:
    pass

  @abstractmethod
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
    sorted_parents = sorted(self.valid_parents.items(),  # Only use tested genes
                            key=lambda params_kv: params_kv[1].fitness, reverse=True)  # Sort by fitness
    worst_params_name = sorted_parents[-1][0]
    del self.pool[worst_params_name]  # Remove from pool

  # Select parents (for breeding) from pool based on fitness
  # Inputs: None
  # Outputs: list of Parameters (self.num_parents long)
  def select_parents(self) -> list[Parameters]:
    params_list = list(self.valid_parents.values())
    fitness_scores = [params.fitness for params in params_list]
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]

    max_ind = np.argmax(probabilities)
    for i, p in enumerate(probabilities):
      if p == 0:
        probabilities[i] += 1e-5
        probabilities[max_ind] -= 1e-5

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

  # End condition for run
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


# class Plateau_Genetic_Algorithm(Genetic_Algorithm):
#
#   # - plateau_sensitivity: How steep the fitness curve must be to be considered a plateau
#   #         Smaller values -> more sensitive to plateau
#   # - plateau_sample_size: How many past fitness's to observe for plateau
#   #         Smaller values -> less accurate detection of plateau
#   # - iterations_per_epoch: Max number of genes to test before starting new epoch
#   # - epochs: Max number of epochs to run before stopping
#   # - Note: iterations is not used for logic in this algorithm (but still needed for constructor).
#   #         replaced with iterations_per_epoch
#   def __init__(self,
#                num_params: int,
#                genome: Genome,
#                epochs: int,
#                iterations_per_epoch: int,
#                warmup: int,
#                plateau_sample_size: int,
#                mutation_decay: float = 1,
#                plateau_sensitivity: float = 5e-5,
#                num_parents: int = 2,
#                past_n_fitness: list = None,
#                founders_pool: dict = None,
#                **kwargs):
#     super().__init__(num_params, -1, genome, num_parents, **kwargs)
#     self.plateau_sensitivity = plateau_sensitivity
#     self.plateau_sample_size = plateau_sample_size
#     self.iterations_per_epoch = iterations_per_epoch
#     self.epochs = epochs
#     self.warmup = warmup
#     self.past_n_fitness = past_n_fitness if past_n_fitness is not None else (np.ones(self.plateau_sample_size) * -np.inf)
#     self.current_epoch = kwargs.pop('current_epoch', 0)   # Not an argument that needs to be set by user
#     self.epoch_iter = kwargs.pop('epoch_iter', 0)
#     self.founders_pool = founders_pool if founders_pool is not None else {}
#
#   def fetch_params(self, **kwargs):
#     self.current_iter += 1  # Total iteration
#     self.epoch_iter += 1    # Iteration for current epoch
#
#     # If pool is unitialized
#     # Initialize new params, add to pool, and return
#     if len(self.pool.items()) < self.num_params:
#       new_params = self.spawn(self.current_iter)
#       new_params.set_attributes(founder_proximity_penalty=self.founder_proximity_penalty(new_params))
#       params_name = get_pool_key(new_params)  # np.array alone cannot be used as key in dict
#       while params_name in self.pool.keys():  # Keep attempting until unique
#         new_params = self.spawn(self.current_iter)
#         new_params.set_attributes(founder_proximity_penalty=self.founder_proximity_penalty(new_params))
#         params_name = get_pool_key(new_params)
#       self.pool[params_name] = new_params
#       return params_name, True
#
#     # If there aren't at least 2 tested Parameters in pool, can't create new Parameters
#     # Return None and unsuccessful flag (False)
#     elif len(self.valid_parents.items()) < 2:
#       self.current_iter -= 1
#       self.epoch_iter -= 1
#       return None, False   # Any changes made during this iteration will be undone
#
#     # Check if max iterations for an epoch
#     # Start new epoch and initialize new params
#     elif self.epoch_iter >= self.iterations_per_epoch:
#       self.start_new_epoch()
#       new_params = self.spawn(self.current_iter)
#       new_params.set_attributes(founder_proximity_penalty=self.founder_proximity_penalty(new_params))
#       params_name = get_pool_key(new_params)
#       self.pool[params_name] = new_params
#       return params_name, True
#
#     ## Add most recent fitness's to list ##
#     # Shift frame for past_n_fitness & add any newly tested genes
#     self.past_n_fitness = np.roll(self.past_n_fitness, -1)
#     for params_key, params in self.valid_parents.items():
#       if params.iteration in range(self.current_iter - self.plateau_sample_size, self.current_iter):
#         ind = self.plateau_sample_size - (self.current_iter - params.iteration)
#         self.past_n_fitness[ind] = params.fitness
#
#     # Check for performance plateau (new epoch if plateau detected & past warmup phase)
#     coefs = np.polyfit(np.arange(len(self.past_n_fitness)), self.past_n_fitness, 1)
#     if coefs[0] < self.plateau_sensitivity and self.epoch_iter > self.warmup:  # If slope is small enough
#       self.start_new_epoch()
#       new_params = self.spawn(self.current_iter)
#       new_params.set_attributes(founder_proximity_penalty=self.founder_proximity_penalty(new_params))
#       params_name = get_pool_key(new_params)
#       self.pool[params_name] = new_params
#       return params_name, True
#
#     # Otherwise, breed new offspring & return
#     else:
#       new_params = self.breed(self.current_iter)
#       params_name = get_pool_key(new_params)    # np.array alone cannot be used as key in dict
#       while params_name in self.pool.keys():  # Keep attempting until unique
#         new_params = self.breed(self.current_iter)
#         params_name = get_pool_key(new_params)
#       self.trim_pool()
#       self.pool[params_name] = new_params
#       return params_name, True
#
#   # Breed new offspring with parents selected from pool
#   # Inputs: current iteration
#   # Outputs: new Parameters (new offspring)
#   def breed(self, iteration: int) -> Parameters:
#     return super().breed(iteration)
#
#   # Create initial Parameters to populate pool
#   # Inputs: current iteration, user-specific keyword args
#   # Outputs: new Parameters
#   def spawn(self, iteration: int, **kwargs) -> Parameters:
#     return super().spawn(iteration, **kwargs)
#
#   # Removes Parameters with lowest fitness from pool
#   # Inputs: None
#   # Outputs: None
#   def trim_pool(self) -> None:
#     super().trim_pool()
#
#   # Select parents (for breeding) from pool based on fitness
#   # Inputs: None
#   # Outputs: list of Parameters (self.num_parents long)
#   def select_parents(self) -> list[Parameters]:
#     return super().select_parents()
#
#   # Crossover parents to create offspring (according to user provided Genome)
#   # Inputs: list of Parameters (parents), current iteration
#   # Outputs: new Parameters (offspring)
#   def crossover(self, parents: list[Parameters], iteration: int) -> Parameters:
#     new_params = super().crossover(parents, iteration)
#     new_params.set_attributes(founder_proximity_penalty=self.founder_proximity_penalty(new_params))
#     return new_params
#
#   # Mutate Parameters (according to user provided Genome)
#   # Inputs: Parameters
#   # Outputs: Parameters (same object, mutated)
#   def mutate(self, params: Parameters) -> Parameters:
#     return super().mutate(params)
#
#   # Begin a new epoch, and return the first params of that epoch
#   def start_new_epoch(self, **kwargs):
#     self.current_epoch += 1
#     self.epoch_iter = 0
#     self.past_n_fitness = (np.ones(self.plateau_sample_size) * -np.inf)
#
#     # Move top scoring paramss to founders pool
#     sorted_params_fitness = sorted(self.valid_parents.items(),
#                      key=lambda params_kv: params_kv[1].fitness + params_kv[1].founder_proximity_penalty, reverse=True)
#     top_params_key, top_params_data = sorted_params_fitness[0]
#     while top_params_key in self.founders_pool.keys():      # Ensure no duplicates
#       sorted_params_fitness = sorted_params_fitness[1:]
#       top_params_key, top_params_data = sorted_params_fitness[0]
#     self.founders_pool[top_params_key] = top_params_data
#
#     # Re-initialize pool
#     # Note: When other model-process's return, fetch_params will handle filling the pool
#     for params_key, params in list(self.valid_parents.items()):
#       del self.pool[params_key]
#
#   # Full override of end_condition. Only end when max epochs reached
#   def end_condition(self):
#     if self.current_epoch >= self.epochs:
#       return True
#
#   # Penalty for being close to founders
#   # Return positive L2 distance between params and all paramss in founders pool
#   def founder_proximity_penalty(self, params: Parameters) -> float:
#     penalty = 0
#     for param_name, param in params.items():
#       penalty += sum([np.linalg.norm(param - founder_param[param_name]) for founder_param in self.founders_pool.values()])
#     return penalty
