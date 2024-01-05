from abc import abstractmethod
from DGA.File_IO import load_gene_file
import numpy as np
import time

POOL_DIR = "pool"
LOCK_DIR = "locks"


class Model():

  # Load data (will only be called with proper filelock)
  def load_data(self, **kwargs):
    pass

  # Write to log
  def logger(self, fitness: float, iteration: int, **kwargs):
    log = {
      "timestamp" : time.strftime('%H:%M:%S', time.localtime()),
      "fitness" : fitness,
      "iteration" : iteration,
    }
    return log

  # Run model
  @abstractmethod
  def run(self, gene, **kwargs) -> float:
    pass

class Testing_Model(Model):
  def __init__(self,
               vector_size: tuple,            # Size of target vectors
               vector_distribution: float,    # Range of target vector values
               num_vectors: int = 1,          # Number of target vectors
               target_vectors: list = None,   # Pre-initialized target vectors
               **kwargs):
    super().__init__(**kwargs)
    self.target_vectors = []

    # Use previously initialized target vectors
    if 'target_vectors' in kwargs:
      self.target_vectors = kwargs['target_vectors']

    # Initialize targets when starting run
    else:

      # Initialize target vectors when passed as args
      if target_vectors is not None:
        self.target_vectors = target_vectors

      # Initialize target vectors randomly
      else:
        for i in range(num_vectors):
          location = np.random.uniform(low=-vector_distribution, high=+vector_distribution)
          self.target_vectors.append(np.random.normal(loc=location, scale=3, size=vector_size))

  # Return distance from closest target vector
  def run(self, gene, **kwargs) -> float:
    smallest_diff = np.inf
    for i, targ in enumerate(self.target_vectors):
      diff = np.linalg.norm(gene.flatten() - targ.flatten())
      if diff < smallest_diff:
        smallest_diff = diff
    return -smallest_diff