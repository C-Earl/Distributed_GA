from DGA.Algorithm import Genetic_Algorithm
from DGA.Client import Client
from DGA.Server import Server
import time

#
# Use the Client class to load your models into DGA. run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
class Simple_Client(Client):                  # <--- Remember to inherit Client class!
  def run(self, gene, **kwargs) -> float:
    fitness = sum([-(i ** 2) for i in gene.flatten()])  # Fitness is (negative) sum of squares
    self.gene = gene.tolist()                 # Save gene as class var. for logging
    return fitness                            # Optimization is to maximize fitness
                                              # Perfectly optimized gene is all zeros

  # By default, 'Client' (base class) tracks fitness and time tested
  # In this override, we add the gene to the log. You can add other info to the log
  # by saving it as a class variable in run(), and updating the log here.
  def logger(self, fitness, **kwargs):
    log = super().logger(fitness, **kwargs)   # Get default log
    log.update({"gene": self.gene})           # Add gene to log
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
  alg = Genetic_Algorithm(gene_shape=(10,10), num_genes=10, mutation_rate=0.05, iterations=1000)
  Server(run_name="my_run",     # Name of run (run files saved in a folder with this name)
         algorithm=alg,           # Algorithm for optimizing your model
         client=Simple_Client(),          # Client class with your model
         num_parallel_processes=5,      # Number of subprocesses to run in parallel)                # Total number of genes to test
         log_pool=5,)                   # Log pool every 1 iterations