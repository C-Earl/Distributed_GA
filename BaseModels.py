import yaml
from abc import ABC, abstractmethod
from typing import Tuple, Union


class Gene(dict):
  def __init__(self,
    id: int,
    fitness: float = -1.0,
    params: dict = None
  ):
    self['fitness'] = fitness
    self['id'] = id
    self['params'] = params
    self.fitness = fitness    # Set as dot notation attributes too (easier to access)
    self.id = id
    self.params = params
    super().__init__()

  def add_fitness(self, fitness: float):
    self.fitness = fitness


class GenePool(dict):
  # Add gene to pool
  def add(self, gene: Gene):
    self[gene['id']] = gene

  def remove(self, gene: Gene):
    self.pop(gene['id'])

  def order_by_fitness(self):
    genes = self.values()
    return sorted(genes, key=lambda g: g['fitness'])


class Model(ABC):
  def __init__(self,
    param_file_path: str,
  ):
    # Load params from YAML file
    with open(param_file_path, 'r') as param_file:
      self.param_skeleton = yaml.safe_load(param_file)

  @abstractmethod
  # Test gene, return fitness (int) and any other data (dict)
  def test(self, gene: Gene, **kwargs) -> Tuple[Union[int, float], dict]:
    pass


class EA(ABC):

  def __init__(self,
    model: Model,  # Model to be trained
    pop_size: int  # Pop size for gene pool
  ):
    self.model = model
    self.gene_pool = GenePool()
    self.pop_size = pop_size
    self.id_count = 0  # Use total genes generated as ID for gene

  @abstractmethod
  # Returns new set of gene created from pool
  # Assumes pool has genes to sample from
  def generate_new_gene(self, **kwargs) -> Gene:
    pass

  @abstractmethod
  # Initialize pool with new genes
  # Different from generate_new_gene() since no genes in pool to sample from
  def init_pool(self, **kwargs):
    pass

  # # Add gene/fitness pair to pool (from client)
  # def add_fitness(self, fitness: int, gene: Gene, **kwargs):
  #   self.gene_pool[gene] = fitness

  # Generate new ID for a gene
  def id_gen(self, **kwargs) -> int:
    new_id = self.id_count
    self.id_count += 1
    return new_id


  @abstractmethod
  # Indicate (to server) that EA is finished executing
  def terminate(self, **kwargs) -> None:
    pass


class Client(ABC):
  def __init__(
    self,
    model: Model,
  ):
    self.model = model

  @abstractmethod
  # Request new gene from EA
  def get_gene(self, **kwargs) -> Gene:
    pass

  @abstractmethod
  # Return fitness for tested gene to EA
  # 'result' is a dict with any additional return data for EA
  def return_gene(self,
                  fitness: int,
                  result: dict = None,
                  ):
    pass

  @abstractmethod
  # Test model on client (+ any other client-side executions)
  def run(self, **kwargs):
    pass


class Server(ABC):
  def __init__(self, alg: EA):
    self.alg = alg

  @abstractmethod
  # Return new gene from EA to Client
  def return_gene(self, gene: Gene, **kwargs):
    pass

  @abstractmethod
  # Insert new fitness/gene pair
  def insert_gene(self, fitness: int, gene: Gene, **kwargs):
    pass

  @abstractmethod
  # Open server for clients to connect to (event-loop goes here)
  def run(self, **kwargs):
    pass
