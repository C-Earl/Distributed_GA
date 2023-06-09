from abc import abstractmethod
from pool_functions import load_gene

POOL_DIR = "pool"
LOCK_DIR = "locks"


class Client():
  def __init__(self, run_name: str, gene_name: str):
    self.run_name = run_name
    self.gene_name = gene_name
    self.gene_data = load_gene(gene_name, run_name)

  # Run model
  @abstractmethod
  def run(self):
    pass
