from DGA.Algorithm import Genetic_Algorithm
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
