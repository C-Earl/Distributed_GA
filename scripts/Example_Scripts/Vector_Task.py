from DGA.Model import Testing_Model
from DGA.Gene import Gene, Genome, uniform_initialization, uniform_mutation, splice_crossover, mean_crossover, splice_mutation
from DGA.Algorithm import Genetic_Algorithm
from DGA.Local import Synchronized
from DGA.Plotting import plot_model_logs
from matplotlib import pyplot as plt


if __name__ == '__main__':
  # Gene: A subsection of the Genome (see below). Represents functional/logical subset of a models parameters.
  #       - Intended to contain single np.ndarray of any shape
  # Example: A gene could represent the weights of a single layer in a neural network.
  simple_gene = Gene(shape=(10,10))

  # Genome: A collection of Genes. Represents full parameter-space of a model.
  #         - You may specify how parameters are initialized, mutated, and crossed-over during reproduction.
  #         Use importable classes like uniform_initialization, uniform_mutation, splice_crossover, etc.
  genome = Genome(initializer=uniform_initialization(min_val=-10, max_val=10),
                  mutator=uniform_mutation(min_val=-1, max_val=1, mutation_rate=0.9),
                  crosser=splice_crossover())
  genome.add_gene(simple_gene, 'simple_gene')

  # Genetic_Algorithm: Class that maintains the pool of tested parameters, and uses the provided Genome to create new
  #                    Parameters for testing.
  alg = Genetic_Algorithm(num_params=10,     # Number of Parameters to keep in pool
                          iterations=1_000,  # Number of unique genes to test
                          genome=genome,)

  # Testing_Model: Class that defines the model to be tested. This is an example model where the goal is to find a
  #                vector of values that is close to a target vector.
  mod = Testing_Model(genome=genome,
                      vector_distribution=10,  # Range of potential target vector locations
                      vector_scale=3)          # Variance in values of target vectors

  # Synchronized: Class that executes the Genetic_Algorithm, and optimizes the Testing_Model.
  synchronized = Synchronized(run_name="Vector_Task", algorithm=alg, model=mod,)
  synchronized.run()

  # Plotting a randomly selected Parameter from the pool against the target vector
  plot_model_logs(run_dir="Vector_Task", num_models=1, )
  test_params = list(synchronized.algorithm.pool.values())[0]['simple_gene']
  closest_targ = mod.target_vectors[0]['simple_gene']
  fig, ax = plt.subplots(1, 2)
  im1 = ax[0].imshow(test_params, vmax=+10, vmin=-10, cmap='bwr')
  ax[0].set_title("Estimate")
  ax[1].imshow(closest_targ, vmax=+10, vmin=-10, cmap='bwr')
  ax[1].set_title(f"Target Vector")
  plt.colorbar(im1, ax=fig.get_axes(), orientation='horizontal')
  plt.show()
