from DGA.Algorithm import Genetic_Algorithm
from DGA.Client import Client
from DGA.Server import Server
import numpy as np


# # # # # # # # # # # # # # # # # # # # # #
# Use the Client class to load your own models into DGA. the run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
class Simple_GA_Client(Client):  # <--- Remember to inherit Client class

  # Description:
  # The function called to test your model. Only requirement is that it returns a float value representing fitness.
  # Gene data is stored in self.gene_data, which is a dictionary with the following keys:
  #   'gene': The gene itself, which is a numpy array
  #   'fitness': The fitness of the gene, which is a float
  #   'status': The status of the gene, which is a string
  #   'time': The time the gene was created, which is a float
  def run(self) -> float:
    gene = self.gene_data['gene']
    fitness = sum([-(i ** 2) for i in gene])
    return fitness

if __name__ == '__main__':
  import os

  alg_path = os.path.abspath(__file__)
  client_path = os.path.abspath(__file__)

  Server(run_name="example_run_name",
         algorithm=Genetic_Algorithm,
         client=Simple_GA_Client,
         num_parallel_processes=5, gene_shape=(10,), num_genes=10, mutation_rate=0.1,
         iterations=20, sbatch_script="run_server.sh")
