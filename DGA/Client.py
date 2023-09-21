from abc import abstractmethod
from DGA.pool_functions import load_gene_file
import numpy as np
import time

POOL_DIR = "pool"
LOCK_DIR = "locks"


class Client():

  # Load data (will only be called with proper filelock)
  def load_data(self, **kwargs):
    pass

  # Write to log
  def logger(self, fitness: float, **kwargs):
    log = {
      "timestamp" : time.strftime('%H:%M:%S', time.localtime()),
      "fitness" : fitness,
    }
    return log

  # Run model
  @abstractmethod
  def run(self, gene, **kwargs) -> float:
    pass
