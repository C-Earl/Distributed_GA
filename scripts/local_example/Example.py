from DGA.Algorithm import Algorithm
from DGA.Client import Client
from DGA.Server import Server
import numpy as np

# # # # # # # # # # # # # # # # # # # # #
# SEE README.md                   			
#          
# Steps to test your own models:
# 1. Create new class that inherits Client class
# 2. Implement run method which runs model and returns fitness value. This is where you can setup your AI-gym, Torch,
#    Tensorflow, etc. models and environments for fitness testing. (details below)
#                              
# Steps to create custom algorithms:			
# 1. Create the class, and inherit the Algorithm object  
# 2. Implement the fetch_gene method which handles the creation of new genes (more below)
# 3. Implement the test_gene method which handles the testing of genes
# # # # # # # # # # # # # # # # # # # # #

class Simple_GA_Client(Client):			# <--- Remember to inherit Client class

	# Description: 
  # The function called to test your model. Only requirement is that it returns a float value representing fitness.
	# Gene data is stored in self.gene_data, which is a dictionary with the following keys:
	#   'gene': The gene itself, which is a numpy array
	#   'fitness': The fitness of the gene, which is a float
	#   'status': The status of the gene, which is a string
	#   'time': The time the gene was created, which is a float
  def run(self) -> float:
    gene = self.gene_data['gene']
    fitness = sum([-(i**2) for i in gene])
    return fitness


class Simple_GA(Algorithm):

  def fetch_gene(self, **kwargs):

    # Only use tested parents
    valid_parents = {gene_key: gene_data for gene_key, gene_data in self.pool.items()  # Filter untested genes
                     if (not gene_data['status'] == 'being tested')}

    # If pool is unitialized, add new gene (phase 1)
    if len(self.pool.items()) < self.num_genes:
      new_gene = np.random.rand(10)
      gene_name = self.create_gene(new_gene)
      return gene_name, True

    # If more than half of the pool is untested, wait.
    elif len(valid_parents.items()) < (self.num_genes / 2):
      return None, False

    # Otherwise, drop lowest fitness and create new gene (phase 2)
    else:

      # Drop lowest fitness
      sorted_parents = sorted(valid_parents.items(), key=lambda gene_kv: gene_kv[1]['fitness'],
                              reverse=True)  # Sort by fitness
      worst_gene = sorted_parents[-1][0]
      self.delete_gene(worst_gene)  # Remove from file dir
      del self.pool[worst_gene]  # Remove from pool obj
      del valid_parents[worst_gene]  # Remove from pool obj

      # Select parents for reproduction
      fitness_scores = [gene_data['fitness'] for _, gene_data in valid_parents.items()]  # Get fitness's (unordered)
      normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
      probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
      p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)
      p1_gene, p2_gene = sorted_parents[p1_i][1]['gene'], sorted_parents[p2_i][1]['gene']

      # Generate offspring with crossover
      crossover_point = np.random.randint(0, self.gene_shape[0])
      new_gene = np.concatenate((p1_gene[:crossover_point], p2_gene[crossover_point:]))

      # Random mutation
      if np.random.rand() < 0.5:
        mutation_point = np.random.randint(0, self.gene_shape[0])
        new_gene[mutation_point] += np.random.uniform(-self.mutation_rate, +self.mutation_rate)

      # new_gene = np.random.rand(10)
      gene_name = self.create_gene(new_gene)
      return gene_name, True

  # Normalize values to positive range [0, +inf) (fitnesses)
  # Do nothing if already in range [0, +inf)
  def pos_normalize(self, values):
    min_v = min(values)
    if min_v < 0:
      return [i + abs(min_v) for i in values]
    else:
      return values


if __name__ == '__main__':
  import os
  alg_path = os.path.abspath(__file__)
  client_path = os.path.abspath(__file__)

  Server(run_name="example_run_name",
         algorithm_path="scripts/local_example/Example.py", algorithm_name="Simple_GA",
         client_path="scripts/local_example/Example.py", client_name="Simple_GA_Client",
         num_parallel_processes=5, gene_shape=(10,), num_genes=10, mutation_rate=0.1,
         iterations=20, sbatch_script="run_server.sh")