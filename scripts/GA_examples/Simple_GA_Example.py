from DGA.Algorithm import Genetic_Algorithm
from DGA.Model import Model
from DGA.Server import Server
import time

#
# Use the Model class to load your models into DGA. run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
class Simple_Model(Model):                  # <--- Remember to inherit Model class!
  def run(self, gene, **kwargs) -> float:
    fitness = sum([-(i ** 2) for i in gene.flatten()])  # Fitness is (negative) sum of squares
    self.gene = gene.tolist()                 # Save gene as class var. for logging
    return fitness                            # Optimization is to maximize fitness
                                              # Perfectly optimized gene is all zeros

  # By default, 'Model' (base class) tracks fitness and time tested
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
#   model: Model class with your model
#   num_parallel_processes: Number of subprocesses to run in parallel
#   **kwargs: Any additional parameters, including args specific to your algorithm.
#             Args are passed automatically to the algorithm and model classes, and
#             can be accessed with self.name_of_kwarg.
# # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
  alg = Genetic_Algorithm(gene_shape=(100,100),		# Shape of gene np.ndarray
                          num_genes=25,						# Number of genes to test
                          mutation_rate=0.25,			# Probability of mutation
                          iterations=100,)				# Total # of genes to test
  Server(run_name="Simple_GA_Example",  	
         algorithm=alg,
         model=Simple_Model(),
         num_parallel_processes=5,)	