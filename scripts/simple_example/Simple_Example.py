from DGA.Algorithm import Genetic_Algorithm_Base, Genetic_Algorithm
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


if __name__ == '__main__':
  import os

  alg_path = os.path.abspath(__file__)
  client_path = os.path.abspath(__file__)

  Server(run_name="simple_example",
         algorithm=Genetic_Algorithm,
         client=Simple_Client,
         num_parallel_processes=5, gene_shape=(10,), num_genes=10, mutation_rate=0.25,
         iterations=20)
