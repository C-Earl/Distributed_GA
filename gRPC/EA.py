from abc import ABC
from Models import Model, Gene
from dataclasses import dataclass
import numpy as np


class GenePool(dict):
  def __init__(self):
    self.id_map = {}  # Map for object ID to object value
    super().__init__()

  # Add gene to pool
  def __setitem__(self, gene: Gene, fitness: [float, int]):
    self.id_map[hash(gene)] = gene
    super().__setitem__(hash(gene), fitness)

  # Get gene-object value from its ID
  def id_to_gene(self, key: int):
    return self.id_map[key]


class EA(object):

  def __init__(self,
    pool_size: int,     # Max number of genes in genepool
    model: Model        # Model to be trained
  ):
    self.model = model

    ### Initialize gene pool ###
    param_skeleton = model.param_skeleton
    self.gene_pool = GenePool()
    for i in range(pool_size):

      # Iterate over all params to create gene
      new_gene = Gene()
      for param_k, param_specs in param_skeleton.items():
        p_min, p_max, p_type, p_shape = param_specs.values()
        new_param = np.random.uniform(p_min, p_max, p_shape).astype(p_type)
        new_gene[param_k] = new_param

      self.gene_pool[new_gene] = 0   # add new gene to pool, fitness 0

  def generate_new_genes(self):
    param_skeleton = self.model.param_skeleton

    # New gene: weighted avg. of gene pool params
    all_fits = np.array([fitness for fitness in self.gene_pool.values()])
    new_gene = Gene()
    if min(all_fits) == max(all_fits):    # for initialization (when fitness are all 0)
      for param_k, param_specs in param_skeleton.items():
        all_params = np.array([self.gene_pool.id_to_gene(gene_key)[param_k] for gene_key in self.gene_pool.keys()])
        avg = np.average(all_params, axis=0)
        new_gene[param_k] = avg
    else:
      for param_k, param_specs in param_skeleton.items():
        weights = (all_fits - min(all_fits)) / (max(all_fits) - min(all_fits))
        all_params = np.array([self.gene_pool.id_to_gene(gene_key)[param_k] for gene_key in self.gene_pool.keys()])
        weighted_avg = np.average(all_params, weights=weights, axis=0)
        new_gene[param_k] = weighted_avg
    return new_gene

  def add_fitness(self, gene: Gene, fitness: int):
    self.gene_pool[gene] = fitness
    print(f"fitness added: {fitness}")
    return 1


if __name__ == '__main__':
  model = Model(param_file_path="../params.yaml")
  EA = EA(10, model)
