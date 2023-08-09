from DGA.Algorithm import Genetic_Algorithm
from DGA.Client import Client
from DGA.Server import Server

#
# Use the Client class to load your models into DGA. run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
class Simple_Client(Client):                  # <--- Remember to inherit Client class!
  def run(self, gene, **kwargs) -> float:
    fitness = sum([-(i ** 2) for i in gene])  # Fitness is (negative) sum of squares
    return fitness                            # Optimization is to maximize fitness
                                              # Perfectly optimized gene is all zeros

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
  import os

  alg_path = os.path.abspath(__file__)
  client_path = os.path.abspath(__file__)

  Server(run_name="simple_example",     # Name of run (run files saved in a folder with this name)
         algorithm=Genetic_Algorithm,   # Algorithm for optimizing your model
         client=Simple_Client,          # Client class with your model
         num_parallel_processes=5,      # Number of subprocesses to run in parallel
         iterations=20,                 # Number of genes each subprocess will test
         gene_shape=(10,), num_genes=10, mutation_rate=0.25)    # Algorithm parameters
