from DGA.Algorithm import Genetic_Algorithm
from DGA.Gene import Gene, Genome, Parameters

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

  alg = Genetic_Algorithm(num_params=10, iterations=1000, genome=genome)
  model = Testing_Model(genome=genome, vector_distribution=3)
  runner = Synchronized(run_name='test', algorithm=alg, model=model)
  runner.run()

  plot_model_logs("test", num_models=1)

