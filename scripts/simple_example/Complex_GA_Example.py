from DGA.Algorithm import Complex_Genetic_Algorithm
from DGA.Client import Client
from DGA.Server import Server
from DGA.Plotting import plot_client_logs
import numpy as np

#
# Use the Client class to load your models into DGA. run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
vector_size = 10
target_vector_1 = np.identity(vector_size)
target_vector_2 = np.flip(np.identity(vector_size), axis=1)
target_vector_3 = np.zeros_like(target_vector_1)
class Simple_Client(Client):                  # <--- Remember to inherit Client class!
  def run(self, gene, **kwargs) -> float:
    smallest_diff = np.inf
    for targ in [target_vector_1, target_vector_2, target_vector_3]:
      diff = np.linalg.norm(gene.flatten()-targ.flatten())
      if diff < smallest_diff:
        smallest_diff = diff
    return -smallest_diff

  # By default, 'Client' (base class) tracks fitness and time tested
  # In this override, we add the gene to the log. You can add other info to the log
  # by saving it as a class variable in run(), and updating the log here.
  def logger(self, fitness, iteration, **kwargs):
    log = super().logger(fitness, iteration, **kwargs)   # Get default log
    # log.update({"gene": self.gene})           # Add gene to log
    return log

#
# The Server class is used to run the genetic algorithm.
# Arguments:
#   run_name: Name of run (run files saved in a folder with this name)
#   algorithm: Algorithm for optimizing your model
#   client: Client class with your model
#   num_parallel_processes: Number of subprocesses to run in parallel
#   iterations: Number of genes each subprocess will test
#   **kwargs: Any additional parameters, including args specific to your algorithm.
#             Args are passed automatically to the algorithm and client classes, and
#             can be accessed with self.name_of_kwarg.
# # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
  alg = Complex_Genetic_Algorithm(gene_shape=(vector_size,vector_size),
                                  num_genes=100,
                                  mutation_rate=0.9,
                                  iterations_per_epoch=10_000,
                                  epochs=5,
                                  plateau_sensitivity=1e-3,
                                  plateau_sample_size=3000,)
  Server(run_name="Complex_GA_Example",     # Name of run (run files saved in a folder with this name)
         algorithm=alg,           # Algorithm for optimizing your model
         client=Simple_Client(),          # Client class with your model
         num_parallel_processes=5,      # Number of subprocesses to run in parallel)                # Total number of genes to test
         log_pool=-1,)                   # Log pool every 1 iterations