from DGA.Algorithm import Complex_Genetic_Algorithm
from DGA.Model import Model
from DGA.Server import Server
from DGA.Plotting import plot_model_logs
import numpy as np

# Generate target vectors
vector_size = 10
target_vector_1 = np.identity(vector_size)
target_vector_2 = np.flip(np.identity(vector_size), axis=1)
target_vector_3 = np.zeros_like(target_vector_1)

# Find closest target vector and return negative distance as fitness
class Simple_Model(Model):                  # <--- Remember to inherit Model class!
  def run(self, gene, **kwargs) -> float:
    smallest_diff = np.inf
    for targ in [target_vector_1, target_vector_2, target_vector_3]:
      diff = np.linalg.norm(gene.flatten()-targ.flatten())
      if diff < smallest_diff:
        smallest_diff = diff
    return -smallest_diff


if __name__ == '__main__':
  # - plateau_sensitivity: How steep the fitness curve must be to be considered a plateau
  #         Smaller values -> more sensitive to plateau
  # - plateau_sample_size: How many past fitness's to observe for plateau
  #         Smaller values -> less accurate detection of plateau
  # - iterations_per_epoch: Max number of genes to test before starting new epoch
  # - epochs: Max number of epochs to run before stopping
  alg = Complex_Genetic_Algorithm(gene_shape=(vector_size,vector_size),
                                  num_genes=100,
                                  mutation_rate=0.9,
                                  iterations_per_epoch=10_000,
                                  epochs=5,
                                  plateau_sensitivity=1e-3,
                                  plateau_sample_size=3000,)
  Server(run_name="Complex_GA_Example",
         algorithm=alg,
         model=Simple_Model(),
         num_parallel_processes=5,
         log_pool=-1, )