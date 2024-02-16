from abc import abstractmethod
from DGA.Gene import Parameters, Genome
import numpy as np
import time

POOL_DIR = "pool"
LOCK_DIR = "locks"

# https://stackoverflow.com/questions/17075071/is-there-a-python-method-to-access-all-non-private-and-non-builtin-attributes-of
# Helper function to get all public attributes of an object
def list_public_attributes(input_var):
  return [k for k, v in vars(input_var).items() if
          not (k.startswith('_') or callable(v))]

class Model():

  def __init__(self):
    self.log_vars = ["timestamp", "fitness", "iteration"]    # List of variables to log
    super().__init__()

  # Load data (will only be called with proper filelock)
  def load_data(self, **kwargs):
    pass

  # Write to log
  def logger(self, params: Parameters):
    # log = {
    #   "timestamp" : time.strftime('%H:%M:%S', time.localtime()),
    #   "fitness" : params.fitness,
    #   "iteration" : params.iteration,
    # }
    log = {}
    # if 'timestamp' in self.log_vars:
    #   log.update({'timestamp': time.strftime('%H:%M:%S', time.localtime())})
    for key in self.log_vars:
      if hasattr(params, key):    # Check attributes
        log[key] = getattr(params, key)
      elif key in params:         # Check actual parameters
        log[key] = params[key]
    # log.update({key: getattr(params, key) for key in self.log_vars})
    return log

  # Run model
  @abstractmethod
  def run(self, params: Parameters, **kwargs) -> float:
    pass

  # Return json-friendly version of model args
  def args_to_json(self) -> dict:
    return {key: self.__dict__[key] for key in list_public_attributes(self)}


class Testing_Model(Model):
  def __init__(self,
               genome: Genome,  # Determines size of parameters
               vector_distribution: float,  # Range of potential target vector locations
               vector_scale: float = 3,  # Variance in values of target vectors
               num_vectors: int = 1,  # Number of target vectors
               target_vectors: list = None,  # Pre-initialized target vectors
               **kwargs: object):
    super().__init__()
    self.target_vectors = []
    self.genome = genome
    self.vector_distribution = vector_distribution

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
    param = list(params.values())[0]    # Only need 1 param, rest ignored
    smallest_diff = np.inf
    for i, targ in enumerate(self.target_vectors):
      for name, gene in self.genome.items():
        if gene.shape is not None:
          # print(params, self.target_vectors)
          targ = targ[name]
        elif self.genome.shape is not None:
          targ = targ
        else:
          raise Exception("Gene or genome defined without shape")
      diff = np.linalg.norm(param.flatten() - targ.flatten())
      if diff < smallest_diff:
        smallest_diff = diff
    return -smallest_diff

  def args_to_json(self) -> dict:
    # json = super().args_to_json()
    json_vectors = []
    for vector in self.target_vectors:
      json_vector = {}
      for name, v in vector.items():
        json_vector[name] = v.tolist()
      json_vectors.append(json_vector)
    json = {
      'target_vectors': json_vectors,
      'vector_distribution': self.vector_distribution,
      'genome': self.genome.to_json()
    }
    return json

  # def json_to_args(self, json: dict) -> dict:
  #   args = {
  #     'target_vectors': json['target_vectors'],
  #     'vector_distribution': json['vector_distribution'],
  #     'genome': Genome.from_json(json['genome'])
  #   }