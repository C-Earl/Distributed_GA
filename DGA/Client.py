from abc import abstractmethod
from DGA.pool_functions import load_gene

POOL_DIR = "pool"
LOCK_DIR = "locks"


class Client():
  def __init__(self, run_name: str, gene_name: str, **kwargs):
    self.run_name = run_name
    self.gene_name = gene_name

  # Run model
  @abstractmethod
  def run(self, gene, **kwargs) -> float:
    pass
