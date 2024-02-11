from typing import List

from DGA.Gene import Gene, Genome, Parameters
import numpy as np

class Hananel_Genome(Genome):
  def __init__(self):
    super().__init__()

  def add_gene(self, gene: Gene, name: str):
    self[name] = gene

  # Called when a new Parameters is needed, and no other Parameters in pool
  # Inputs: iteration
  # Outputs: new Parameters
  def initialize(self, iteration: int) -> Parameters:
    return super().initialize(iteration)

  # Takes in a Parameters object and mutates it (Note: should return *SAME* Parameters object, just modified)
  # Inputs: Parameters
  # Outputs: Parameters (mutated)
  def mutate(self, params: Parameters) -> Parameters:
    return super().mutate(params)

  # Takes in a Parameters object and crosses it with another Parameters object
  # Inputs: list of Parameters (parents)
  # Outputs: Parameters (offspring)
  def crossover(self, parents: List[Parameters], iteration: int) -> Parameters:
    return super().crossover(parents, iteration)

  ###  IMPLEMENT ME  ###
  # Notes for Hananel:
  # 1. Use this file/class to add your own mutation/crossover methods. You can access these functions
  # from Hananel_Algorithm.py by calling:      self.genome.<function_name>(<args>)
  #
  # 2. Check DGA.Gene (the super functions used in crossover & mutate) for examples on how to use these
  def merge_mutate(self, params: Parameters) -> Parameters:
    # Use params.as_array() to get a flattened, np.ndarray of the parameters
    flat_param = params.as_array()
    pass

  ###  IMPLEMENT ME  ###
  def multipoint_mutate(self, params: Parameters) -> Parameters:
    flat_param = params.as_array()
    pass
