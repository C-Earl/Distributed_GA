from DGA.Algorithm import Genetic_Algorithm_Base
from DGA.Gene import Gene, Genome, Parameters
from os.path import join as file_path
from DGA.Pool import Pool, Subset_Pool
from DGA.File_IO import load_gene_file as load_param_file
from abc import abstractmethod
import os
import numpy as np

POOL_DIR = "pool"
POOL_LOCK_NAME = "POOL_LOCK.lock"


# Returns unique key for a Parameters object
# Used as key in pool for Algorithm
def get_pool_key(params: Parameters):
  return hash(params)


class Updated_Genetic_Algorithm_Base:
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

    # File management vars
    self.run_name = kwargs.pop('run_name')
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = Pool()  # Subset_Pool() will have all elements of Pool() meeting condition
    self.valid_parents = Subset_Pool(condition=lambda key, params: params.tested())
    self.pool.add_subset_pool(self.valid_parents)

    # Load pool
    self.pool.extend(self.load_pool())

  def load_pool(self):
    pool = {}
    for root, dirs, files in os.walk(self.pool_path):
      for file in files:
        file_name = file.split('.')[0]  # This will be unique hash of the param
        params = load_param_file(self.run_name, file_name)
        pool[file_name] = params
    return pool

  # Fetch a new Parameters from pool. Called when a new params needed for testing.
  # Inputs: None
  # Outputs: Parameters and success status (whether new Parameters were created, returns (None, False) if not)
  @abstractmethod
  def fetch_params(self) -> tuple[Parameters, bool]:
    pass

  # Create new Parameters using provided Genome (self.genome) and current pool (self.pool)
  # Inputs: None
  # Outputs: new Parameters object
  @abstractmethod
  def breed(self) -> Parameters:
    pass

  # Create new Parameters from scratch (no pool)
  # Inputs: None
  # Outputs: new Parameters object
  @abstractmethod
  def spawn(self, iteration: int) -> Parameters:
    pass

  # Trim pool of weaker Parameters
  # Inputs: None
  # Outputs: None
  @abstractmethod
  def trim_pool(self) -> None:
    pass

  # Select new parents from pool based on fitness
  # Inputs: None
  # Outputs: tuple of Parameter objects (parents)
  @abstractmethod
  def select_parents(self) -> tuple:
    pass

  # Crossover parents to create offspring
  # Inputs: list of Parameter objects (parents), current iteration (set as new Parameters iteration)
  # Outputs: new Parameters object (offspring)
  @abstractmethod
  def crossover(self, parents: list[Parameters], iteration: int) -> Parameters:
    pass

  # Mutate Parameters
  # Inputs: Parameters object
  # Outputs: Parameters object (same object)
  @abstractmethod
  def mutate(self, params: Parameters) -> Parameters:
    pass

  # End condition for run. Returns True if end condition met, otherwise False
  # Inputs: None
  # Outputs: bool (True if end condition met)
  @abstractmethod
  def end_condition(self) -> bool:
    pass


class Updated_Genetic_Algorithm(Updated_Genetic_Algorithm_Base):
  def __init__(self,
               num_params: int = -1,
               iterations: int = -1,
               genome: Genome = None,
               num_parents: int = 2,
               **kwargs) -> None:
    super().__init__(num_params, iterations, genome, **kwargs)
    self.num_parents = num_parents  # TODO: Implement this

  # Fetch a new Parameters from pool for testing.
  def fetch_params(self) -> tuple:
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
      new_params = self.breed()
      params_name = get_pool_key(new_params)  # np.array alone cannot be used as key in dict
      while params_name in self.pool.keys():  # Keep attempting until unique
        new_params = self.breed()
        params_name = get_pool_key(new_params)

      # Remove worst Parameters from the pool
      self.trim_pool()

      # Add new Parameters to pool & return
      self.pool[params_name] = new_params
      return params_name, True

  def breed(self) -> Parameters:
    parents = self.select_parents()
    offspring = self.crossover(parents, self.current_iter)
    offspring = self.mutate(offspring)
    return offspring

  def spawn(self, iteration: int) -> Parameters:
    return self.genome.initialize(iteration)

  # Removes Parameters with lowest fitness from pool
  def trim_pool(self) -> None:
    sorted_parents = sorted(self.valid_parents.items(),  # Only use tested genes
                            key=lambda params_kv: params_kv[1].fitness, reverse=True)  # Sort by fitness
    worst_params_name = sorted_parents[-1][0]
    del self.pool[worst_params_name]  # Remove from pool

  def select_parents(self) -> list[Parameters]:
    params_list = list(self.valid_parents.values())
    fitness_scores = [params.fitness for params in params_list]
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
    parent_inds = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=self.num_parents)
    return [params_list[i] for i in parent_inds]

  def crossover(self, parents: list[Parameters], iteration: int) -> Parameters:
    return self.genome.crossover(parents, iteration)

  def mutate(self, params: Parameters) -> Parameters:
    return self.genome.mutate(params)

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

if __name__ == '__main__':
  from DGA.Gene import Gene, Genome, Parameters
  from DGA.Model import Testing_Model
  from DGA.Local import Synchronized
  from DGA.Plotting import plot_model_logs
  import numpy as np

  def initialize(shape) -> np.ndarray:
    return np.random.normal(loc=0, scale=1, size=shape)

  def crossover(parents: list[np.ndarray]) -> np.ndarray:
    return np.mean(parents, axis=0)

  def mutate(param: np.ndarray, shape) -> np.ndarray:
    param += np.random.normal(loc=0, scale=1, size=shape)
    return param

  genome = Genome()
  genome.add_gene('test_gene', Gene(shape=(10, 10), datatype=float, initialize=initialize, crossover=crossover, mutate=mutate))

  alg = Updated_Genetic_Algorithm(num_params=10, iterations=100, genome=genome)
  model = Testing_Model(vector_size=(10, 10), vector_distribution=3)
  runner = Synchronized(run_name='test', algorithm=alg, model=model)
  runner.run()

  plot_model_logs("test", num_models=1)

