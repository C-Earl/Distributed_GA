from abc import abstractmethod
from DGA.pool_functions import load_gene

POOL_DIR = "pool"
LOCK_DIR = "locks"


class Client():

  # Load data (will only be called with proper filelock)
  def load_data(self, **kwargs):
    pass

  # Run model
  @abstractmethod
  def run(self, gene, **kwargs) -> float:
    pass
