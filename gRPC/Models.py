from abc import abstractmethod
import numpy as np
import yaml


# 'Gene' is modified dict that is immutable and hashable
# Note: Objects id() is immutable & constant during its lifetime
class Gene(dict):
  def __hash__(self):
    return id(self)


class Model(object):

  def __init__(
    self,
    param_file_path: str,
  ):
    # Load params from YAML file
    with open(param_file_path, 'r') as param_file:
      self.param_skeleton = yaml.safe_load(param_file)

    # Simple array to guess
    self.params_to_guess = np.random.rand(50)

  def test(self, gene: Gene):
    test_params = gene["TEST_PARAMS"]
    return 1/np.linalg.norm(test_params - np.arange(start=0, stop=10))
