from DGA.Algorithm import Genetic_Algorithm_Base as Algorithm
from DGA.Client import Client


class Synchronized:
  def __init__(self, run_name: str, algorithm: Algorithm, client: Client | type, genes_per_iter: int, **kwargs):
    self.run_name = run_name
    self.algorithm = algorithm
    self.client = client
    self.genes_per_iter = genes_per_iter

  def run(self):
    # Loop until end condition met
    while not self.algorithm.end_condition():
      for i in range(self.genes_per_iter):
        new_gene = self.algorithm.fetch_gene()
        fitness = self.client.run(new_gene)

    pass