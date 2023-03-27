import os
import pickle
import numpy as np
import yaml
from os import mkdir, system as cmd
import subprocess
from os.path import join as join_path
from typing import Tuple, Union
from abc import ABC, abstractmethod

from BaseModels import Server, Client, EA, Model, Gene, GenePool


# Gene
class TestGene(Gene):
  params: np.array

def gene_representer(dumper, data):
  values = {'id': data.id, 'fitness': data.fitness, 'params': data.params}
  return dumper.represent_mapping('!TestGene', values)
yaml.add_representer(TestGene, gene_representer)

def gene_constructor(loader, node):
  values = loader.construct_mapping(node, deep=True)
  return Gene(**values)
yaml.add_constructor('!TestGene', gene_constructor, Loader=yaml.UnsafeLoader)

def write_to_yaml(gene, file: str):
  with open(file, 'w') as yaml_file:
    yaml.dump(gene, yaml_file)

def read_from_yaml(file: str):
  with open(file, 'r') as yaml_file:
    gene = yaml.load(yaml_file, Loader=yaml.UnsafeLoader)
  return gene

# File meta data
CLIENT_BASE_DIR = "clients"
CLIENT_FILE = lambda client_num: f"client_{client_num}"
POOL_DIR = "gene_pool"
GENE_FILE = lambda g_id: f"gene_{g_id}.yaml"
ID_FILE = "id_count.txt"

class SlurmServer(Server):
  def __init__(self,
    alg: EA,
    sim_dir: str,  # Path to simulation directory (all things written here)
    num_clients: int,  # Num. nodes to run on
    client_setup: str,  # Path to bash script to setup run env. on node
    verbose: bool = False,
    initial_run: bool = True,  # True if initializing, False if callback from client
    gene_id: int = None, # Should be provided if init is False
    local: bool = False, # TODO: TEMPORARY
  ):
    super().__init__(alg)   # Adds alg as class var
    self.sim_dir = sim_dir
    self.client_setup = client_setup
    self.num_clients = num_clients
    self.verbose = verbose
    self.base_client_dir = join_path(sim_dir, CLIENT_BASE_DIR)
    self.base_pool_dir = join_path(sim_dir, POOL_DIR)

    # Do normal run if this is callback
    if not initial_run:
      self.run()
      return

    # TODO: Potentially remove this
    # Initialize Client file-system
    # if verbose: print("Initializing client file system ...")
    # mkdir(self.sim_dir)
    # mkdir(self.base_client_dir)
    # for i in range(num_clients):
    #   mkdir(join_path(self.base_client_dir, CLIENT_FILE(i)))   # Indiv. client dir's

    # Request nodes (training starts)
    #
    # LOCAL CALL IS TEMPORARY:
    # Execution will run on event-loop here in constructor. Asynchronous subprocesses called
    # to execute client code (run model, etc.), and resynchronized every loop iteration.
    if local:
      if verbose: print("Queueing local nodes...")
      popens = []
      for i in range(num_clients):
        popens.append(subprocess.Popen(["bash", f"./{client_setup}", str(i), str(int(local))]))
      for p in popens:
        p.wait()

      # After init genes tested, move to loop
      if verbose: print("Init finished, moving to run()...")
      while not ea.terminate():
        for i in range(num_clients):
          self.run(local=local)             # Run server code...
        if verbose: print("Queueing local nodes...")
        popens = []
        for i in range(num_clients):        # Run clients...
          popens.append(subprocess.Popen(["bash", f"./{client_setup}", str(i), str(int(local))]))
        for p in popens:
          p.wait()


    else:
      if verbose: print("Queueing nodes with sbatch...")
      for i in range(num_clients):
        cmd(f"sbatch ./{client_setup} {i}")

  # Return new gene from EA to Client
  def return_gene(self, gene: Gene, **kwargs):
    gene_id = gene['id']
    write_to_yaml(gene, join_path(self.base_pool_dir, GENE_FILE(gene_id)))

  # Add fitness to gene in pool
  def insert_gene(self, fitness: int, gene: Gene, **kwargs):
    pass      # Not necessary since client writes directly to gene

  # Handle callback's from Client: New gene + new node for testing
  def run(self, **kwargs):
    # Extract fitness from gene   NOT NEEDED because client writes directly to gene
    # if self.verbose: print("Extracting fitness...")
    # gene = read_from_yaml(join_path(self.base_pool_dir, GENE_FILE(gene_id)))
    # fitness = gene.fitness

    # Add fitness to pool         NOT NEEDED because client writes directly to gene
    # self.insert_gene(fitness, gene)

    # Allocate new gene to client
    if self.verbose: print("Allocating new gene...")
    new_gene = self.alg.generate_new_gene()
    self.return_gene(new_gene)


    # TODO: TEMPORARY
    # Return to __init__
    if kwargs["local"]:
      return


    # Sbatch new node
    if self.verbose: print("Queueing new node with sbatch...")
    # cmd(f"bash ./{self.client_setup} {gene_id}")


class SlurmClient(Client):
  def __init__(self,
    model: Model,
    sim_dir: str,           # Where client will get data from
    gene_id: int,           # Gene from pool this client will test
    server_callback: str,   # Path to bash script to callback Server
    verbose: bool = False,
  ):
    super().__init__(model)
    self.verbose = verbose
    self.sim_dir = sim_dir
    self.gene_id = gene_id
    self.callback = server_callback

  # Retrieve gene from disk
  def get_gene(self, **kwargs):
    return read_from_yaml(join_path(self.sim_dir, POOL_DIR, GENE_FILE(self.gene_id)))

  # Write fitness to disk
  def return_gene(self,
                  fitness: int,
                  results: dict = None,
                  ):
    # Read gene from file
    gene = read_from_yaml(join_path(self.sim_dir, POOL_DIR, GENE_FILE(self.gene_id)))

    # Update gene fitness and rewrite to file
    gene['fitness'] = fitness
    gene['results'] = results
    write_to_yaml(gene, join_path(self.sim_dir, POOL_DIR, GENE_FILE(self.gene_id)))

  # Test model & write results to disk
  def run(self, **kwargs):

    # Test gene
    if self.verbose: print("Testing genes...")
    gene = self.get_gene()
    fitness, results = self.model.test(gene=gene)

    # Write fitness to disk
    self.return_gene(fitness, results)


    # TEMPORARY: If a local run, finish and return to server
    if kwargs['local']:
      return


    # Callback to Server for new gene
    cmd(f"bash ./{self.callback} {self.gene_id}")


class TestModel(Model):
  def __init__(self, param_file_path: str):
    super().__init__(param_file_path)

  def test(self, gene: TestGene, **kwargs) -> Tuple[Union[int, float], dict]:
    num_arr = gene['params']['num_arr']
    return float(1 / np.sum(num_arr - np.array([0,1,2,3,4,5,6,7,8,9]))), {}


class SlurmEA(EA, ABC):

  def __init__(self, model: Model, pop_size: int, sim_dir: str):
    super().__init__(model, pop_size)
    self.sim_dir = sim_dir
    self.base_pool_dir = join_path(sim_dir, POOL_DIR)
    self.id_file = join_path(ID_FILE)

    # Generate pool files and genes (if not there yet)
    if not os.path.isdir(self.sim_dir):
      os.mkdir(self.sim_dir)        # Make directories
      os.mkdir(self.base_pool_dir)
      with open(self.id_file, 'w') as id_file:
        id_file.write("0")
      self.id_count = 0

      # Initialize pool & write genes to files
      self.init_pool()
      for gene in self.gene_pool.values():
        write_to_yaml(gene, join_path(self.base_pool_dir, GENE_FILE(gene.id)))


  @abstractmethod
  # Handle EA logic here
  def generate_new_gene(self, **kwargs) -> TestGene:
    pass

  @abstractmethod
  # Initialize gene pool with custom gene TestGene
  def init_pool(self, **kwargs):
    pass

  @abstractmethod
  # Indicate to server that EA is done
  def terminate(self, **kwargs) -> bool:
    pass

  # Write id to sim file
  def id_gen(self) -> int:
    new_id = super().id_gen()
    with open(self.id_file, 'w') as id_file:
      id_file.write(str(self.id_count))
    return new_id

  # Get genes from files
  def load_genes(self):
    for gene_file in os.listdir(self.base_pool_dir):
      gene = read_from_yaml(gene_file)
      self.gene_pool.add(gene)


class TestEA(SlurmEA):
  # Handle EA logic here
  def generate_new_gene(self, **kwargs) -> TestGene:
    # Weighted avg of top 5 gene performing genes
    sorted_genes = self.gene_pool.order_by_fitness()[0:5]
    sorted_params = [gene['params']['num_arr'] for gene in sorted_genes]
    sorted_fitness = [gene['fitness'] for gene in sorted_genes]
    weighted_avg = np.average(sorted_params, weights=sorted_fitness, axis=0)
    new_params = weighted_avg + np.random.rand(len(weighted_avg))         # Random mutation

    # Remove lowest performing gene
    weakest_gene = self.gene_pool.order_by_fitness()[-1]
    self.gene_pool.remove(weakest_gene)
    os.remove(join_path(self.base_pool_dir, GENE_FILE(weakest_gene.id)))

    new_gene = TestGene(id=self.id_gen(), params={'num_arr' : new_params.tolist()})
    self.gene_pool.add(new_gene)
    return new_gene

  # Initialize gene pool with custom gene TestGene
  def init_pool(self, **kwargs):
    for i in range(self.pop_size):
      new_gene = TestGene(id=self.id_gen(), params={'num_arr': np.random.uniform(0, 10, size=10).tolist()})
      self.gene_pool.add(new_gene)

  # Indicate to server that EA is done
  def terminate(self, **kwargs) -> bool:
    top_gene = self.gene_pool.order_by_fitness()[0]
    return top_gene.fitness > 10

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-server", type=int)
  parser.add_argument("-initial_run", type=int)
  parser.add_argument("-id", type=int, default=-1)
  parser.add_argument("-local", type=int)
  parser.add_argument("-num_clients", type=int, default=2)
  args = parser.parse_args()
  server = bool(args.server)
  initial_run = bool(args.initial_run)
  gid = args.id
  local = bool(args.local)
  num_clients = args.num_clients

  sim_dir = "test_sim"
  ## Run server ##
  if server:
    # If doing initial run, run a setup
    if initial_run:
      import shutil
      try:
        shutil.rmtree(sim_dir)          # Clear previous sim files
        os.remove('id_count.txt')
      except FileNotFoundError:
        pass

      # Define model and EA
      model = TestModel("params.yaml")
      ea = TestEA(model, 10, sim_dir)

      # Begin simulation...
      SlurmServer(ea, sim_dir, num_clients, "client_setup.sh", verbose=True, initial_run=True, local=local)

    # If this is a callback from a client, requeue node with run()
    else:
      # Recreate server
      model = TestModel("params.yaml")
      ea = TestEA(model, 10, sim_dir)
      SlurmServer(ea, sim_dir, num_clients, "client_setup.sh", verbose=True, initial_run=False, gene_id=gid, local=local)

  ## Run client ##
  else:
    import logging
    import traceback

    # Create a logging object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    # Create a file handler
    handler = logging.FileHandler('error.log')

    # Set the logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(handler)

    try:
      model = TestModel("params.yaml")
      client = SlurmClient(model, sim_dir="test_sim", gene_id=args.id, server_callback="server_callback.sh")
      client.run(verbose=True, local=local)
    except Exception as e:
      logger.error("An error occurred: %s", traceback.format_exc())
