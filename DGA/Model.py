from abc import abstractmethod
from DGA.Gene import Parameters, Genome
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
               genome: Genome,                # Determines size of parameters
               vector_distribution: float,    # Range of target vector values
               vector_scale: float = 3,       # Scale of target vector values
               num_vectors: int = 1,          # Number of target vectors
               target_vectors: list = None,   # Pre-initialized target vectors
               **kwargs):
    super().__init__(**kwargs)
    self.target_vectors = []
    self.genome = genome

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
          new_target = {}
          for name, gene in genome.items():
            # Get vector size for each gene
            if gene.shape is not None:
              vector_size = gene.shape
            elif genome.shape is not None:
              vector_size = genome.shape
            else:
              raise Exception("Gene or genome defined without shape")

            # Initialize target vector
            location = np.random.uniform(low=-vector_distribution, high=+vector_distribution)
            new_target[name] = np.random.normal(loc=location, scale=vector_scale, size=vector_size)
          self.target_vectors.append(new_target)

  # Return distance from closest target vector
  def run(self, params: Parameters, **kwargs) -> float:
    param = params.values()[0]    # Only need 1 param, rest ignored
    smallest_diff = np.inf
    for i, targ in enumerate(self.target_vectors):
      for name, gene in self.genome.items():
        if gene.shape is not None:
          targ = targ[name]
        elif self.genome.shape is not None:
          targ = targ
        else:
          raise Exception("Gene or genome defined without shape")
      diff = np.linalg.norm(param.flatten() - targ.flatten())
      if diff < smallest_diff:
        smallest_diff = diff
    return -smallest_diff