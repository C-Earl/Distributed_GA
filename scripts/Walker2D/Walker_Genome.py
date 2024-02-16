from DGA.Gene import Genome, Gene, Parameters
import numpy as np

class Walker_Genome(Genome):

  # Called when a new Parameters is needed, and no other Parameters in pool
  # Inputs: iteration
  # Outputs: new Parameters
  def initialize(self, iteration: int) -> Parameters:
    new_params = Parameters(iteration=iteration)
    for gene_name, gene in self.items():
      gdefault = gene.default
      if gdefault is not None:    # If default value is provided, use it
        new_params[gene_name] = gdefault
      else:
        if gene.dtype == bool:
          new_params[gene_name] = np.random.choice([True, False], size=gene.shape)
        else:
          gshape = gene.shape       # Otherwise, uniform generate values in gene range
          gmin = gene.min_val
          gmax = gene.max_val
          gtype = gene.dtype
          new_params[gene_name] = np.random.uniform(low=gmin, high=gmax, size=gshape).astype(gtype)
    return new_params

  # Takes in a Parameters object and mutates it (Note: Returns same Parameters object)
  # Inputs: Parameters
  # Outputs: Parameters (mutated)
  def mutate(self, params: Parameters) -> Parameters:
    for gene_name, gene in self.items():
      gshape = gene.shape
      gtype = gene.dtype      # Apply uniform mutation to each gene
      params[gene_name] += np.random.uniform(low=-1, high=+1, size=gshape).astype(gtype)
    return params

  # Takes in a Parameters object and crosses it with another Parameters object
  # Inputs: list of Parameters (parents)
  # Outputs: Parameters (offspring)
  def crossover(self, parents: list[Parameters], iteration: int) -> Parameters:
    p1, p2 = parents[0], parents[1]  # Only two parents used for now, change later
    child_params = Parameters(iteration=iteration)
    for gene_name, gene in self.items():
      gshape = p1[gene_name].shape
      full_index = np.prod(gshape)
      splice = np.random.randint(low=0, high=full_index)
      new_param = np.concatenate([p1[gene_name].flatten()[:splice], p2[gene_name].flatten()[splice:]])
      child_params[gene_name] = new_param.reshape(gshape)
    return child_params