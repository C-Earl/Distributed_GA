from DGA.Algorithm import Genetic_Algorithm
from DGA.Client import Client
from DGA.Server_SLURM import Server_SLURM

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
  def logger(self, fitness, iteration, **kwargs):
    log = super().logger(fitness, iteration, **kwargs)   # Get default log
    return log

#
# The Server class is used to run the genetic algorithm.
# Arguments:
#   run_name: Name of run (run files saved in a folder with this name)
#   algorithm: Algorithm for optimizing your model
#   client: Client class with your model
#   num_parallel_processes: Number of subprocesses to run in parallel
#   **kwargs: (WIP)
# # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
  alg = Genetic_Algorithm(gene_shape=(100,100),
                          num_genes=25,
                          mutation_rate=0.25,
                          iterations=100,)
  Server_SLURM(run_name="Simple_GA_Example",     # Name of run (run files saved in a folder with this name)
         algorithm=alg,           			# Algorithm for optimizing your model
         sbatch_script="run_client.sh",  # Script to run when creating clients for runs
         client=Simple_Client(),          # Client class with your model
         num_parallel_processes=5,)      # Number of subprocesses to run in parallell 
