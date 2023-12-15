from DGA.Algorithm import Plateau_Genetic_Algorithm
from DGA.Model import Model
from DGA.Local import Synchronized
from DGA.Plotting import plot_model_logs
import numpy as np

#
# Use the Model class to load your models into DGA. run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
class Vector_Estimation_Model(Model):  # <--- Remember to inherit Model class!
  def __init__(self, num_vectors: int, vector_size: tuple, vector_distribution: float, target_vectors: list = None, **kwargs):
    super().__init__(**kwargs)

    # Initialize target vectors
    self.target_vectors = []

    # Use previously initialized target vectors
    if 'target_vectors' in kwargs:
      self.target_vectors = kwargs['target_vectors']
    else:                             # Initialize targets when starting run

      # Initialize target vectors when passed as args
      if target_vectors is not None:
        self.target_vectors = target_vectors

      # Initialize target vectors randomly
      else:
        for i in range(num_vectors):
          location = np.random.uniform(low=-vector_distribution, high=+vector_distribution)
          self.target_vectors.append(np.random.normal(loc=location, scale=1, size=vector_size))

  def run(self, gene, **kwargs) -> float:
    smallest_diff = np.inf
    for i, targ in enumerate(self.target_vectors):
      diff = np.linalg.norm(gene.flatten() - targ.flatten())
      if diff < smallest_diff:
        smallest_diff = diff
    return -smallest_diff

  def logger(self, fitness, iteration, **kwargs):
    log = super().logger(fitness, iteration, **kwargs)  # Get default log
    return log


class Custom_Plateau_Algorithm(Plateau_Genetic_Algorithm):

  def initial_gene(self, **kwargs):
    return np.random.normal(loc=0, scale=10, size=self.gene_shape)

  # Return mutated gene
  def mutate(self, gene, **kwargs):
    if np.random.rand() < self.mutation_rate:
      mutation_start = np.random.randint(0, np.prod(self.gene_shape))
      mutation_end = np.random.randint(mutation_start+1, mutation_start+2+(self.mutation_rate*5))
      if mutation_end > np.prod(self.gene_shape):
        mutation_end = np.prod(self.gene_shape)
      gene = gene.flatten()
      gene[mutation_start:mutation_end] += np.random.normal(loc=0, scale=self.mutation_rate*5, size=mutation_end-mutation_start)
    return gene.reshape(self.gene_shape)

  def founder_proximity_penalty(self, gene):
    return super().founder_proximity_penalty(gene) / 7


#
# The Server class is used to run the genetic algorithm.
# Arguments:
#   run_name: Name of run (run files saved in a folder with this name)
#   algorithm: Algorithm for optimizing your model
#   model: Model class with your model
#   num_parallel_processes: Number of subprocesses to run in parallel
#   iterations: Number of genes each subprocess will test
#   **kwargs: Any additional parameters, including args specific to your algorithm.
#             Args are passed automatically to the algorithm and model classes, and
#             can be accessed with self.name_of_kwarg.
# # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
  import matplotlib.pyplot as plt

  num_epochs = 10
  alg = Custom_Plateau_Algorithm(gene_shape=(vector_size, vector_size),
                                 num_genes=10,
                                 mutation_rate=1,
                                 mutation_decay=.99995,
                                 iterations_per_epoch=15_000,
                                 epochs=num_epochs,
                                 plateau_sensitivity=1e-5,
                                 plateau_sample_size=2000,
                                 warmup=2000)
  mod = Vector_Estimation_Model()
  sync = Synchronized(run_name="Local_Complex_GA_Example",  # Name of run (run files saved in a folder with this name)
         algorithm=alg,  # Algorithm for optimizing your model
         model=mod,)  # Model class with your model

  sync.run()
  plot_model_logs("Local_Complex_GA_Example", num_models=1)
  for _, gene_data in alg.founders_pool.items():
    # Find which target vector the gene is closest to
    smallest_diff = np.inf
    for i, targ in enumerate([mod.target_vector_1, mod.target_vector_2, mod.target_vector_3]):
      diff = np.linalg.norm(gene_data['gene'].flatten() - targ.flatten())
      if diff < smallest_diff:
        smallest_diff = diff
        closest_targ = targ
        closest_targ_id = i

    # Plot gene
    fig, ax = plt.subplots(1, 2)
    im1 = ax[0].imshow(gene_data['gene'], vmax=1, vmin=-1, cmap='bwr')
    ax[0].set_title("Gene")
    ax[1].imshow(closest_targ, vmax=1, vmin=-1, cmap='bwr')
    ax[1].set_title(f"Closest Target Vector {closest_targ_id}")
    plt.colorbar(im1, ax=fig.get_axes(), orientation='horizontal')

    plt.show()
