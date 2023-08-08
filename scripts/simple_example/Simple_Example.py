from DGA.Algorithm import Genetic_Algorithm
from DGA.Client import Client
from DGA.Server import Server
import numpy as np


# # # # # # # # # # # # # # # # # # # # # #
# Use the Client class to load your own models into DGA. the run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
class Simple_Client(Client):  # <--- Remember to inherit Client class
  def run(self, gene, **kwargs) -> float:
    fitness = sum([-(i ** 2) for i in gene])
    return fitness


class Simple_GA(Genetic_Algorithm):  # <--- Remember to inherit Genetic_Algorithm class

  # Initialize with random values
  # Called automatically if pool is not full
  def initial_gene(self, **kwargs):
    return np.random.rand(10)

  # Remove worst gene from pool
  def remove_weak(self, gene_pool: dict):
    sorted_parents = sorted(gene_pool.items(),
                            key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)  # Sort by fitness
    worst_gene = sorted_parents[-1][0]
    del gene_pool[worst_gene]  # Remove from pool obj
    return gene_pool

  # Weighted selection of parents based on fitness
  def select_parents(self, gene_pool: dict):
    fitness_scores = [gene_data['fitness'] for _, gene_data in gene_pool.items()]  # Get fitness's (unordered)
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
    p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)
    sorted_genes = sorted(gene_pool.items(), key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)
    return sorted_genes[p1_i][1]['gene'], sorted_genes[p2_i][1]['gene']

  # Crossover parents at random point
  def crossover(self, p1, p2):
    crossover_point = np.random.randint(0, self.gene_shape[0])
    new_gene = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
    return new_gene

  # Mutate gene at random point
  def mutate(self, gene):
    if np.random.rand() < 0.5:
      mutation_point = np.random.randint(0, self.gene_shape[0])
      gene[mutation_point] += np.random.uniform(-self.mutation_rate, +self.mutation_rate)
    return gene

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

  Server(run_name="simple_example",
         algorithm=Simple_GA,
         client=Simple_Client,
         num_parallel_processes=5, gene_shape=(10,), num_genes=10, mutation_rate=0.1,
         iterations=20)
