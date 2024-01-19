from DGA.Model import Testing_Model
from DGA.Algorithm import Genetic_Algorithm, get_pool_key
from DGA.Gene import Genome, Gene, Parameters, base_mutate_class, base_crossover_class, uniform_mutation, \
  uniform_initialization, mean_crossover, splice_crossover
from DGA.Local import Synchronized
import numpy as np


class merge_mutation(base_mutate_class):
  def __init__(self, mutation_rate: float = 0.1):
    super().__init__(mutation_rate)

  def mutate(self, gene: Gene) -> Gene:
    pass


class multi_points_mutation(base_mutate_class):
  def __init__(self, mutation_rate: float = 0.1):
    super().__init__(mutation_rate)

  def mutate(self, gene: Gene) -> Gene:
    pass


class custom_crossover(base_crossover_class):
  def __init__(self):
    super().__init__()

  def crossover(self, gene1: Gene, gene2: Gene) -> Gene:
    pass


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
               cross_points: int = 1,  # Optional: Number of crossover points if provided
               plateau_range: int = 100,  # Optional: Number of iterations to check for plateau
               **kwargs):
    super().__init__(num_params, -1, genome, num_parents, **kwargs)    # Don't use default 'self.iterations', set -1
    self.diversity_threshold = diversity_threshold
    self.cross_points = cross_points
    self.iterations_per_epoch = iterations_per_epoch
    self.epochs = epochs
    self.current_epoch = 0
    self.epoch_iter = 0   # Iteration within current epoch
    self.total_iter = 0   # Total iterations
    self.plateau_range = plateau_range
    self.founders_pool = {}  # Pool of top scoring Parameters from previous epochs
    # self.diversity_matrix = np.zeros((self.num_params, self.num_params))  # Distance between all params
    self.fitness_history = np.zeros((self.num_params, plateau_range))     # Record of past fitness (per agent)

  # Fetch a new Parameters from pool for testing.
  # Inputs: None
  # Outputs: tuple of (params_name, success_flag)
  def fetch_params(self, **kwargs) -> tuple:
    # TODO: GA_class.py line 581 to 671, merge_mutation, multi_points_mutation, custom_crossover defined above
    self.epoch_iter += 1
    self.total_iter += 1

    # If epoch is over
    if self.epoch_iter >= self.iterations_per_epoch:
      self.start_new_epoch()
      new_params = self.spawn(self.total_iter)
      params_name = get_pool_key(new_params)
      self.pool[params_name] = new_params
      return params_name, True

    # Generate new Parameters normally
    else:
      params_name, status = super().fetch_params()
      self.pool[params_name].iteration = self.total_iter  # Adjust iteration to match total
      return params_name, status

  def founder_proximity_penalty(param: Parameters, founder_pool: dict) -> float:
    return -1

  # Begin a new epoch, and return the first params of that epoch
  def start_new_epoch(self, **kwargs):
    self.current_epoch += 1
    self.epoch_iter = 0
    self.fitness_history = np.zeros((self.num_params, self.plateau_range))  # Reset fitness history

    # Move top scoring params to founders pool
    sorted_params = sorted(self.valid_parents.items(), key=lambda x: x[1].fitness, reverse=True)
    top_param_name, top_param = sorted_params[0]
    self.founders_pool[top_param_name] = top_param

    # Re-initialize pool
    # Note: When other model-process's return, fetch_params will handle filling the pool
    for param_key, param in list(self.valid_parents.items()):
      del self.pool[param_key]

  # Breed new offspring with parents selected from pool
  # Inputs: current iteration
  # Outputs: new Parameters (new offspring)
  def breed(self, iteration: int):
    parents = self.select_parents()
    offspring = self.crossover(parents, iteration)
    offspring = self.mutate(offspring)
    return offspring

  # Removes Parameters with lowest fitness from pool
  # Inputs: None
  # Outputs: None
  def trim_pool(self):
    # TODO: When do you trim?
    # TODO: How do you trim?
    super().trim_pool()
    pass

  # Select parents (for breeding) from pool based on fitness
  # Inputs: None
  # Outputs: list of Parameters (self.num_parents long)
  def select_parents(self) -> list[Parameters]:
    # TODO: Does this use weighted probability by fitness?
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
    # TODO: Where is end condition?
    if self.current_epoch == self.epochs:
      return True

  # Check an agents history for plateau-ing (convergence)
  def agent_converged(self, agent_id: int) -> bool:
    score = np.zeros((self.num_params,))

  def pos_normalize(self, values):
    min_v = min(values)
    if min_v < 0:
      return [i + abs(min_v) for i in values]
    else:
      return values

if __name__ == '__main__':
  # Run variables
  VECTOR_SHAPE = (10, 10)

  # Genome
  genome = Genome()
  gene = Gene(shape=VECTOR_SHAPE,
              initializer=uniform_initialization(min_val=-10, max_val=10),
              mutator=uniform_mutation(min_val=-1, max_val=1, mutation_rate=0.9),
              crosser=splice_crossover())
  genome.add_gene(gene, 'vector_gene')

  mod = Testing_Model(genome=genome, vector_size=VECTOR_SHAPE, vector_distribution=10, vector_scale=3)
  alg = Hananel_Algorithm(genome=genome, num_params=10, iterations_per_epoch=1000, epochs=5)
  sync_runner = Synchronized(run_name="Hananel_Alg", algorithm=alg, model=mod)
  sync_runner.run()

  from DGA.Plotting import plot_model_logs
  plot_model_logs(run_dir="Hananel_Alg", num_models=1)