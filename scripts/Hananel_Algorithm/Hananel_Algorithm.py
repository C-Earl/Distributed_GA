from DGA.Algorithm import Genetic_Algorithm, get_pool_key
from DGA.Gene import Genome, Gene, Parameters
from Hananel_Genome import Hananel_Genome
from scipy.spatial import distance
import numpy as np


class Hananel_Algorithm(Genetic_Algorithm):
  def __init__(self,
               # Genetic Algorithm Parameters
               genome: Hananel_Genome,  # Genome to use for creating new Parameters
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

    self.diversity_history = []  # History of diversity scores
    self.diversity_matrix = np.zeros((self.num_params, self.num_params))  # Distance between all params
    self.agent_iterations = np.zeros(self.num_params, dtype=int)          # Iterations per agent (per epoch)
    self.mutation_rate = 0.1    # Dynamically adjusted mutation rate

  # Fetch a new Parameters from pool for testing.
  # Inputs: None
  # Outputs: tuple of (params_name, success_flag)
  def fetch_params(self, **kwargs) -> tuple:
    agent_id = kwargs['agent_id']
    self.epoch_iter += 1
    self.total_iter += 1
    self.agent_iterations[agent_id] += 1

    # If pool initialized, start calculating diversity matrix
    if not (len(self.pool.items()) < self.num_params):
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
    if len(self.pool.items()) < self.num_params:
      new_params = self.spawn(self.total_iter)
      params_name = str(hash(new_params))
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
      penalty += sum(
        [np.linalg.norm(param - founder_param[param_name]) for founder_param in self.founders_pool.values()])
    return -penalty

  # Begin a new epoch, and return the params first params of that epoch
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
    # Create new offspring
    parents = self.select_parents()
    offspring = self.genome.crossover(parents, iteration)

    # Apply mutation with 'mutation rate'% chance
    if np.random.rand() < self.mutation_rate:
      offspring = self.genome.mutate(offspring)

    offspring.as_array()

    ### IMPLEMENT ME ###
    # Apply merge_mutate, multipoint_mutate, etc. here
    # Access genome functions with: self.genome.<function_name>(<args>)
    # Example: self.genome.merge_mutate(offspring)

    return offspring

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

    # Get (normalized) diversity scores
    normalized_dmat = (self.diversity_matrix - self.diversity_matrix.min()) / (
        self.diversity_matrix.max() - self.diversity_matrix.min())  # Normalize to [0, 1]
    param_diversities = np.array([param_div.mean() for param_div in normalized_dmat])

    ### IMPLEMENT ME ###
    # Apply age penalty here
    # 'Age' of Parameters stored as 'iteration' attribute: params.iteration
    # Sorting pool params by iteration:

    # Calculate probabilities
    probabilities = normed_fitness + param_diversities  # Normalize to [0, 1]
    probabilities /= np.sum(normed_fitness + param_diversities)
    max_ind = np.argmax(probabilities)
    for i, p in enumerate(probabilities):
      if p == 0:
        probabilities[i] += 1e-5
        probabilities[max_ind] -= 1e-5
    parent_inds = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities,
                                   size=self.num_parents)
    return [params_list[i] for i in parent_inds]

  # End condition for run. Ends when all epochs complete
  # Inputs: None
  # Outputs: bool (True if run should end)
  def end_condition(self):
    if self.current_epoch == (self.epochs-1) and self.epoch_iter == (self.iterations_per_epoch-1):
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
      return True     #  If slope of fitness curve is less than threshold, return True
    else:
      return False    # Otherwise, return False

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
