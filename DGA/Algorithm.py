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
from scipy.spatial import distance

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
               pool_size: int = -1,  # Number of Parameters in pool
               iterations: int = -1,  # Number of iterations to run
               genome: Genome = None,  # Genome to use for creating new Parameters
               **kwargs) -> None:

    # Algorithm specific vars. Only turned to class vars if defined by user
    if pool_size != -1:
      self.pool_size = pool_size
    if iterations != -1:
      self.iterations = iterations
    if genome is not None:
      self.genome = genome
    self.current_iter = kwargs.pop('current_iter', 0)  # Not an argument that needs to be set by user

    # Check if run from user script or server
    # Note: When users start a run, they must pass a GA object with pool_size & iterations
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

  # Initialize pool
  # Inputs: num_params (int) - Number of Parameters to initialize (CAN be more than pool_size)
  # Outputs: copy of newly initialized pool
  def initialize_pool(self, num_params: int):
    for i in range(num_params):
      new_params = self.spawn(self.current_iter)
      params_name = get_pool_key(new_params)
      self.pool[params_name] = new_params
    return self.pool.copy()


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
               pool_size: int,
               iterations: int,
               genome: Genome,
               num_parents: int = 2,
               **kwargs) -> None:
    super().__init__(pool_size, iterations, genome, **kwargs)
    self.num_parents = num_parents

  # Fetch a new Parameters from pool for testing.
  # Inputs: None
  # Outputs: tuple of (params_name, success_flag)
  def fetch_params(self, **kwargs) -> tuple:
    self.current_iter += 1  # Increment iteration

    # If pool is uninitialized, initialize new Parameters
    # if len(self.pool.items()) < self.pool_size:
    #   new_params = self.spawn(self.current_iter)
    #   params_name = get_pool_key(new_params)
    #   self.pool[params_name] = new_params
    #   return params_name, True

    # If there aren't at least 2 genes in pool, can't create new gene
    if len(self.valid_parents.items()) < 2:
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
    if len(self.valid_parents) > self.pool_size:
      num_to_remove = len(self.valid_parents) - self.pool_size
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


class Hananel_Algorithm(Genetic_Algorithm):
  def __init__(self,
               # Genetic Algorithm Parameters
               genome: Genome,  # Genome to use for creating new Parameters
               pool_size: int,  # Number of Parameters in pool

               # Hananel Algorithm Parameters
               iterations_per_epoch: int,  # Number of iterations per epoch
               epochs: int,  # Number of epochs to run
               num_parents: int = 2,  # Number of parents to select for breeding
               diversity_threshold: float = 0.1,  # Threshold for diversity
               diversity_method: str = 'euclidean',  # Optional: Method for calculating diversity
               cross_points: int = 1,  # Optional: Number of crossover points if provided
               plateau_range: int = 100,  # Optional: Number of iterations to check for plateau
               plateau_sensitivity: float = 1e-5,  # Optional: Slope threshold for plateau detection
               plateau_warmup: int = 0,  # Optional: Number of iterations to ignore before plateau detection
               **kwargs):
    # Don't use default 'self.iterations', set -1
    # History is required for plateau detection
    super().__init__(pool_size, -1, genome, num_parents,
                     history=True, log_vars=['proximity_penalty'], **kwargs)

    self.diversity_threshold = diversity_threshold
    self.diversity_method = diversity_method
    self.cross_points = cross_points
    self.iterations_per_epoch = iterations_per_epoch
    self.epochs = epochs
    self.current_epoch = 0
    self.epoch_iter = 0  # Iteration within current epoch
    self.total_iter = 0  # Total iterations
    self.plateau_range = plateau_range
    self.plateau_sensitivity = plateau_sensitivity
    self.plateau_warmup = plateau_warmup
    self.founders_pool = {}  # Pool of top scoring Parameters from previous epochs

    self.diversity_history = []  # History of diversity scores
    self.diversity_matrix = np.zeros((self.pool_size, self.pool_size))  # Distance between all params
    self.agent_iterations = np.zeros(self.pool_size, dtype=int)  # Iterations per agent (per epoch)

  # Fetch a new Parameters from pool for testing.
  # Inputs: None
  # Outputs: tuple of (params_name, success_flag)
  def fetch_params(self, **kwargs) -> tuple:
    agent_id = kwargs['agent_id']
    self.epoch_iter += 1
    self.total_iter += 1
    self.agent_iterations[agent_id] += 1

    # If pool initialized, start calculating diversity matrix
    if not (len(self.pool.items()) < self.pool_size):
      matrix_pool = np.vstack([params.as_array() for params in self.valid_parents.values()])
      self.diversity_matrix = distance.cdist(matrix_pool, matrix_pool, self.diversity_method)
      # Why "- self.diversity_matrix.shape[0]" ?
      diversity = self.diversity_matrix.sum() / (self.diversity_matrix.size - self.diversity_matrix.shape[0])
      self.diversity_history.append(diversity)
      self.mutation_rate = 0.9

    # Filter out prior epoch Parameters
    # Param already written to file & logged, so remove from pool & generate new init param
    for param_key, param in self.valid_parents.items():
      if param.epoch < self.current_epoch:
        del self.pool[param_key]
        new_params = self.spawn(self.total_iter)
        params_name = str(hash(new_params))
        new_params.set_attribute('agent_id', agent_id)
        new_params.set_attribute('proximity_penalty', self.founder_proximity_penalty(new_params))
        new_params.set_attribute('epoch', self.current_epoch)
        self.pool[params_name] = new_params
        return params_name, True

    # TODO: Make so server handles first two if cases
    # If pool is uninitialized, initialize new Parameters
    if len(self.pool.items()) < self.pool_size:
      new_params = self.spawn(self.total_iter)
      params_name = str(hash(new_params))
      new_params.set_attribute('agent_id', agent_id)
      new_params.set_attribute('proximity_penalty', self.founder_proximity_penalty(new_params))
      new_params.set_attribute('epoch', self.current_epoch)
      self.pool[params_name] = new_params
      return params_name, True

    # If there aren't at least num_parents genes in pool, can't create new gene
    elif len(self.valid_parents.items()) < self.num_parents:
      self.total_iter -= 1  # No gene created, so don't increment (cancels out prior += 1)
      self.epoch_iter -= 1
      return None, False

    # If epoch over (iterations)
    elif self.epoch_iter >= self.iterations_per_epoch or self.agent_converged(agent_id):
      self.start_new_epoch()
      new_params = self.spawn(self.total_iter)
      params_name = str(hash(new_params))
      new_params.set_attribute('agent_id', agent_id)
      new_params.set_attribute('proximity_penalty', self.founder_proximity_penalty(new_params))
      new_params.set_attribute('epoch', self.current_epoch)
      self.pool[params_name] = new_params
      return params_name, True

    # Otherwise, breed new offspring
    else:
      new_params = self.breed(self.total_iter)
      params_name = str(hash(new_params))
      new_params.set_attribute('agent_id', agent_id)
      new_params.set_attribute('proximity_penalty', self.founder_proximity_penalty(new_params))
      new_params.set_attribute('epoch', self.current_epoch)

      # Remove worst Parameters from the pool
      self.trim_pool()

      # Add new Parameters to pool & return
      self.pool[params_name] = new_params
      return params_name, True

  def founder_proximity_penalty(self, params: Parameters) -> float:
    penalty = 0
    for param_name, param in params.items():
      if param.dtype == bool:
        penalty += np.sum([param == founder_param[param_name] for founder_param in self.founders_pool.values()])
      else:
        penalty += sum(
          [np.linalg.norm(param - founder_param[param_name]) for founder_param in self.founders_pool.values()])
    return -penalty

  # Begin a new epoch, and return the first params of that epoch
  def start_new_epoch(self, **kwargs):
    self.current_epoch += 1
    self.epoch_iter = 0
    self.agent_iterations = np.zeros(self.pool_size, dtype=int)  # Reset agent iterations

    # Move top scoring params to founders pool
    sorted_params = self.sort_params(self.valid_parents)
    top_param_name, top_param = sorted_params[0]
    self.founders_pool[top_param_name] = top_param

    # Re-initialize pool
    # Note: Only remove valid_parents
    #       Removing untested genes will cause callback errors for agents testing those genes
    for param_key, param in list(self.valid_parents.items()):
      del self.pool[param_key]

  # Breed new offspring with parents selected from pool
  # Inputs: current iteration
  # Outputs: new Parameters (new offspring)
  def breed(self, iteration: int) -> Parameters:
    # Create new offspring
    parents = self.select_parents()
    offspring = self.genome.crossover(parents, iteration)

    # Apply mutation with 'mutation rate'% chance
    if np.random.rand() < self.mutation_rate:
      offspring = self.genome.mutate(offspring)

    # offspring_array = offspring.as_array()

    ### IMPLEMENT ME ###
    # Apply merge_mutate, multipoint_mutate, etc. here
    # Access genome functions with: self.genome.<function_name>(<args>)
    # Example: self.genome.merge_mutate(offspring)

    # offspring.from_array(offspring_array)

    return offspring

  # Removes Parameters with lowest fitness from pool
  # Inputs: None
  # Outputs: None
  def trim_pool(self) -> None:
    # Handle for when async causes overpopulation
    if len(self.valid_parents) > self.pool_size:
      num_to_remove = len(self.valid_parents) - self.pool_size
      sorted_params = self.sort_params(self.valid_parents)
      for i in range(num_to_remove):
        param_name = sorted_params[-(i + 1)][0]
        del self.pool[param_name]

    # Remove worst params based on fitness
    else:
      sorted_params = self.sort_params(self.valid_parents)
      worst_params_name = sorted_params[-1][0]
      del self.pool[worst_params_name]
    # worst_params_name = sorted_params[-1][0]
    # del self.pool[worst_params_name]  # Remove from pool

  # Select parents (for breeding) from pool based on fitness
  # Inputs: None
  # Outputs: list of Parameters (self.num_parents long)
  def select_parents(self) -> list[Parameters]:
    # Get (normalized) fitness scores
    params_list = list(self.valid_parents.values())
    fitness_scores = [params.fitness + params.proximity_penalty for params in params_list]
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)

    # Get (normalized to [0, 1]) diversity scores
    if self.diversity_matrix.max() == self.diversity_matrix.min():
      # Avoid divide by zero. Logically, all diversity scores are equal in this case
      normalized_dmat = np.zeros(self.pool_size)
    else:
      normalized_dmat = (self.diversity_matrix - self.diversity_matrix.min()) / (
              self.diversity_matrix.max() - self.diversity_matrix.min())
    param_diversities = np.array([param_div.mean() for param_div in normalized_dmat])

    ### IMPLEMENT ME ###
    # Apply age penalty here
    # 'Age' of Parameters stored as 'iteration' attribute: params.iteration
    # Sorting pool params by iteration:

    # Calculate probabilities
    probabilities = normed_fitness + param_diversities  # Normalize to [0, 1]
    # If all probabilities are 0 or contain NaN, set to uniform
    if probabilities.sum() == 0 or (np.isnan(probabilities)).any():
      probabilities = np.ones(self.pool_size) / self.pool_size
    else:
      probabilities /= np.sum(normed_fitness + param_diversities)

    # max_ind = np.argmax(probabilities)
    # for i, p in enumerate(probabilities):
    #   if p == 0:
    #     probabilities[i] += 1e-5
    #     probabilities[max_ind] -= 1e-5

    parent_inds = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities,
                                   size=self.num_parents)
    return [params_list[i] for i in parent_inds]

  # End condition for run. Ends when all epochs complete
  # Inputs: None
  # Outputs: bool (True if run should end)
  def end_condition(self):
    if self.current_epoch == self.epochs:
      return True

  # Check an agents history for plateau-ing (convergence)
  def agent_converged(self, agent_id: int) -> bool:
    # Check if agent has tested enough params
    num_tested = self.agent_iterations[agent_id]
    if num_tested < self.plateau_warmup:
      return False

    # Check for fitness plateaus
    agent_history = self.history[agent_id][-self.plateau_range:]
    fitness_history = np.array([log['fitness'] for log in agent_history])
    coefs = np.polyfit(np.arange(len(fitness_history)), fitness_history, 1)
    if coefs[0] < self.plateau_sensitivity:
      return True  # If slope of fitness curve is less than threshold, return True
    else:
      return False  # Otherwise, return False

  # Takes a dictionary of Parameters, returns a list of tuples containing the key of the Parameter and the Parameter
  def sort_params(self, params_list: dict[str, Parameters]) -> list[tuple[str, Parameters]]:
    sorted_params = sorted(params_list.items(), key=lambda x: x[1].fitness + x[1].proximity_penalty, reverse=True)
    return sorted_params

  def pos_normalize(self, values) -> np.ndarray:
    min_v = min(values)
    if min_v < 0:
      return np.array([i + abs(min_v) for i in values])
    else:
      return values
