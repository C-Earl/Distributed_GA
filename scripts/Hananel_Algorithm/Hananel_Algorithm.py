from DGA.Algorithm import Genetic_Algorithm, get_pool_key
from DGA.Gene import Genome, Gene, Parameters
from scipy.spatial import distance
import numpy as np


class Hananel_Algorithm(Genetic_Algorithm):
  def __init__(self,
               # Genetic Algorithm Parameters
               genome: Genome,  # Genome to use for creating new Parameters
               num_params: int,  # Number of Parameters in pool

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
    super().__init__(num_params, -1, genome, num_parents,
                     history=True, log_vars=['proximity_penalty'], **kwargs)

    self.diversity_threshold = diversity_threshold
    self.diversity_method = diversity_method
    self.cross_points = cross_points
    self.iterations_per_epoch = iterations_per_epoch
    self.epochs = epochs
    self.current_epoch = 0
    self.epoch_iter = 0   # Iteration within current epoch
    self.total_iter = 0   # Total iterations
    self.plateau_range = plateau_range
    self.plateau_sensitivity = plateau_sensitivity
    self.plateau_warmup = plateau_warmup
    self.founders_pool = {}  # Pool of top scoring Parameters from previous epochs
    self.diversity_matrix = np.zeros((self.num_params, self.num_params))  # Distance between all params
    self.fitness_history = np.zeros((self.num_params, plateau_range))     # Record of past fitness (per agent)
    self.agent_iterations = np.zeros(self.num_params, dtype=int)          # Iterations per agent (per epoch)

  # Fetch a new Parameters from pool for testing.
  # Inputs: None
  # Outputs: tuple of (params_name, success_flag)
  def fetch_params(self, **kwargs) -> tuple:
    agent_id = kwargs['agent_id']
    self.epoch_iter += 1
    self.total_iter += 1
    self.agent_iterations[agent_id] += 1
    matrix_pool = np.ndarray([params.as_array() for params in self.valid_parents.values()])
    self.diversity_matrix = distance.cdist(matrix_pool, matrix_pool, self.diversity_method)

    # Filter out prior epoch Parameters
    # Param already written to file & logged, so remove from pool & generate new init param
    for param_key, param in self.valid_parents.items():
      if param.epoch < self.current_epoch:
        del self.pool[param_key]
        new_params = self.spawn(self.total_iter)
        params_name = get_pool_key(new_params)
        new_params.set_attribute('agent_id', agent_id)
        new_params.set_attribute('proximity_penalty', self.founder_proximity_penalty(new_params))
        new_params.set_attribute('epoch', self.current_epoch)
        self.pool[params_name] = new_params
        return params_name, True

    # TODO: Make so server handles first two if cases
    # If pool is uninitialized, initialize new Parameters
    if len(self.pool.items()) < self.num_params:
      new_params = self.spawn(self.total_iter)
      params_name = get_pool_key(new_params)
      new_params.set_attribute('agent_id', agent_id)
      new_params.set_attribute('proximity_penalty', self.founder_proximity_penalty(new_params))
      new_params.set_attribute('epoch', self.current_epoch)
      self.pool[params_name] = new_params
      return params_name, True

    # If there aren't at least 2 genes in pool, can't create new gene
    elif len(self.valid_parents.items()) < 2:
      self.total_iter -= 1  # No gene created, so don't increment (cancels out prior += 1)
      self.epoch_iter -= 1
      return None, False

    # If epoch over (iterations)
    elif self.epoch_iter >= self.iterations_per_epoch or self.agent_converged(agent_id):
      self.start_new_epoch()
      new_params = self.spawn(self.total_iter)
      params_name = get_pool_key(new_params)
      new_params.set_attribute('agent_id', agent_id)
      new_params.set_attribute('proximity_penalty', self.founder_proximity_penalty(new_params))
      new_params.set_attribute('epoch', self.current_epoch)
      self.pool[params_name] = new_params
      return params_name, True

    # Otherwise, breed new offspring
    else:
      new_params = self.breed(self.total_iter)
      params_name = get_pool_key(new_params)
      new_params.set_attribute('agent_id', agent_id)
      new_params.set_attribute('proximity_penalty', self.founder_proximity_penalty(new_params))
      new_params.set_attribute('epoch', self.current_epoch)
      while params_name in self.pool.keys():  # Keep attempting until unique
        new_params = self.breed(self.total_iter)
        params_name = get_pool_key(new_params)

      # Remove worst Parameters from the pool
      self.trim_pool()

      # Add new Parameters to pool & return
      self.pool[params_name] = new_params
      return params_name, True

  def founder_proximity_penalty(self, params: Parameters) -> float:
    penalty = 0
    for param_name, param in params.items():
      penalty += sum(
        [np.linalg.norm(param - founder_param[param_name]) for founder_param in self.founders_pool.values()])
    return -penalty

  # Begin a new epoch, and return theparams first params of that epoch
  def start_new_epoch(self, **kwargs):
    self.current_epoch += 1
    self.epoch_iter = 0
    self.agent_iterations = np.zeros(self.num_params, dtype=int)  # Reset agent iterations

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
    parents = self.select_parents()
    offspring = self.crossover(parents, iteration)
    offspring = self.mutate(offspring)
    return offspring

  # Removes Parameters with lowest fitness from pool
  # Inputs: None
  # Outputs: None
  def trim_pool(self) -> None:
    sorted_params = self.sort_params(self.valid_parents)
    worst_params_name = sorted_params[-1][0]
    del self.pool[worst_params_name]  # Remove from pool

  # Select parents (for breeding) from pool based on fitness
  # Inputs: None
  # Outputs: list of Parameters (self.num_parents long)
  def select_parents(self) -> list[Parameters]:
    # TODO: Change according to age of Parameter (iteration)
    params_list = list(self.valid_parents.values())
    fitness_scores = [params.fitness for params in params_list]
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
    parent_inds = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities,
                                   size=self.num_parents)
    return [params_list[i] for i in parent_inds]

  # End condition for run
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

    # Check for plateaus
    agent_history = self.history[agent_id][-num_tested:]
    fitness_history = np.array([log['fitness'] for log in agent_history])
    coefs = np.polyfit(np.arange(len(fitness_history)), fitness_history, 1)
    if coefs[0] < self.plateau_sensitivity:
      return True     #  If slope of fitness curve is less than threshold, return True
    else:
      return False    # Otherwise, return False

  def sort_params(self, params_list: dict[str, Parameters]) -> list[tuple[str, Parameters]]:
    sorted_params = sorted(params_list.items(), key=lambda x: x[1].fitness + x[1].proximity_penalty, reverse=True)
    return sorted_params

  def pos_normalize(self, values):
    min_v = min(values)
    if min_v < 0:
      return [i + abs(min_v) for i in values]
    else:
      return values


class Hananel_Genome(Genome):
  def __init__(self):
    super().__init__()

  # Called when a new Parameters is needed, and no other Parameters in pool
  # Inputs: iteration
  # Outputs: new Parameters
  def initialize(self, iteration: int) -> Parameters:
    new_params = Parameters(iteration=iteration)
    for gene_name, gene in self.items():
      gdefault = gene.default
      if gdefault is not None:    # If default value is provided, use it
        new_params[gene_name] = gdefault
      else:
        gshape = gene.shape       # Otherwise, uniform generate values in gene range
        gmin = gene.min_val
        gmax = gene.max_val
        gtype = gene.dtype
        new_params[gene_name] = np.random.uniform(low=gmin, high=gmax, size=gshape).astype(gtype)
    return new_params

  # Takes in a Parameters object and mutates it (Note: Returns same Parameters object)
  # Inputs: Parameters
  # Outputs: Parameters (mutated)
  def mutate(self, params: Parameters) -> Parameters:
    for gene_name, gene in self.items():
      gshape = gene.shape
      gtype = gene.dtype      # Apply uniform mutation to each gene
      params[gene_name] += np.random.uniform(low=-1, high=+1, size=gshape).astype(gtype)
    return params

  # Takes in a Parameters object and crosses it with another Parameters object
  # Inputs: list of Parameters (parents)
  # Outputs: Parameters (offspring)
  def crossover(self, parents: list[Parameters], iteration: int) -> Parameters:
    p1, p2 = parents[0], parents[1]  # Only two parents used for now, change later
    child_params = Parameters(iteration=iteration)
    for gene_name, gene in self.items():
      gshape = p1[gene_name].shape
      full_index = np.prod(gshape)
      splice = np.random.randint(low=0, high=full_index)
      new_param = np.concatenate([p1[gene_name].flatten()[:splice], p2[gene_name].flatten()[splice:]])
      child_params[gene_name] = new_param.reshape(gshape)
    return child_params

  # Decay mutation rate (unimplemented)
  def decay_mutators(self):
    pass
