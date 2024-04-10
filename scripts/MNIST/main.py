import numpy as np

from DGA.Gene import Parameters, Genome, Gene
from DGA.Algorithm import Genetic_Algorithm as Algorithm
from DGA.Local import Synchronized
from DGA.Model import Model
from MNIST_Model import MNIST_Model
import torch

class DecayGenome(Genome):
  def __init__(self, rate_decay: float = 0.999, scale_decay: float = 0.999):
    self.rate_decay = rate_decay
    self.scale_decay = scale_decay
    super().__init__()

  # Reduce mutation rate based on iteration of param
  def mutate(self, params: Parameters) -> Parameters:
    for gene_name, gene in self.items():
      gshape = gene.shape
      gtype = gene.dtype
      gmin = gene.min_val
      gmax = gene.max_val
      mut_rate = gene.mutation_rate
      mut_scale = gene.mutation_scale
      rate_modifier = self.rate_decay ** params.iteration
      scale_modifier = self.scale_decay ** params.iteration
      if np.random.uniform() < (mut_rate * rate_modifier):
        if gene.dtype == bool:  # If gene is boolean, flip value
          params[gene_name] = np.logical_not(params[gene_name])
          continue

        if gshape is not None:    # If gene is array, mutate each element
          params[gene_name] += np.random.uniform(low=-(mut_scale*scale_modifier), high=(mut_scale*scale_modifier), size=gshape).astype(gtype)
        else:                # If gene is scalar, mutate the value
          params[gene_name] += np.random.uniform(low=-(mut_scale*scale_modifier), high=(mut_scale*scale_modifier))
        params[gene_name] = np.clip(params[gene_name], gmin, gmax)
    return params


if __name__ == '__main__':
  genome = DecayGenome(rate_decay=0.999, scale_decay=0.999)
  conv1_weight = Gene(shape=(1, 1, 5, 5), dtype=float, min_val=-2, max_val=2, mutation_rate=0.5, mutation_scale=0.1)
  conv1_bias = Gene(shape=(1,), dtype=float, min_val=-2, max_val=2, mutation_rate=0.5, mutation_scale=0.1)
  fc1_weight = Gene(shape=(10, 36), dtype=float, min_val=-2, max_val=2, mutation_rate=0.5, mutation_scale=0.1)
  fc1_bias = Gene(shape=(10,), dtype=float, min_val=-2, max_val=2, mutation_rate=0.5, mutation_scale=0.1)
  genome.add_gene(conv1_weight, 'conv1.weight')
  genome.add_gene(conv1_bias, 'conv1.bias')
  genome.add_gene(fc1_weight, 'fc1.weight')
  genome.add_gene(fc1_bias, 'fc1.bias')

  mod = MNIST_Model()
  alg = Algorithm(pool_size=10, iterations=300, genome=genome, num_parents=2)
  sync_runner = Synchronized(run_name="MNIST_run", algorithm=alg, model=mod)
  sync_runner.run()

  # test_model = MNIST_Model()
  # test_model.load_data()
  # out = test_model.run(test_params)
  # print(out)


# test_params = Parameters(iteration=0)
# test_params.update(torch.load('MNIST_model.pth'))
# test_params['conv1.weight'] = torch.rand(1, 1, 5, 5)
# test_params['conv1.bias'] = torch.rand(1)
# test_params['fc1.weight'] = torch.rand(10, 144)
# test_params['fc1.bias'] = torch.rand(10)