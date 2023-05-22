import time
from os import mkdir, rmdir
from os.path import join as file_path, isdir
import os
import argparse
import pickle
import numpy as np
import datetime
import sys
import subprocess

###################### GLOBALS ######################
POOL_DIR = "pool"
CLIENT_RUNNER = "run_client.sh"
SERVER_CALLBACK = "run_server.sh"
GENE_NAME = lambda cid: f"gene_{cid}"   # cid = client id
###################### GLOBALS ######################

################# HELPER FUNCTIONS ##################
def write_gene(gene: dict, name: str, run_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, name) + ".pkl"
  with open(gene_path, 'wb') as gene_file:
    pickle.dump(gene, gene_file)

def load_gene(name: str, run_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, name) + ".pkl"
  with open(gene_path, 'rb') as gene_file:
    gene = pickle.load(gene_file)
  return gene

# TODO: Make this more robust
def get_pool_key(gene: np.array):
  list_gene = gene.tolist()
  return tuple(list_gene)

def get_gene(pool_key: tuple):
  return np.asarray(pool_key)
################# HELPER FUNCTIONS ##################

# TODO: Delete this stuff
def test_marker(id):
  with open(f"{id}.txt", 'w') as file:
    file.write("hi")

def client_test_marker(cid, count, gene):
  with open(f"{cid}.txt", 'w') as file:
    file.write(str(count))
    file.write(str(gene))

class Model():
  def run(self, gene):
    # Evaluate gene
    return sum(np.array([1,2,3,4,5,6,7,8,9,10]) - gene['gene'])

class Server():
  def __init__(self, run_name: str, model_name: str, num_clients: int, recall: bool = False, **kwargs):
    self.run_name = run_name

    ### RECALL HANDLING ###
    if recall:
      ### Handle kwarg dtypes here ###
      kwargs['num_genes'] = int(kwargs['num_genes'])

      # Run algorithm, write fitness, get new gene, call client to run new gene
      client_id = kwargs['client_id']

      # TODO: Testing code, delete
      count = int(kwargs['count'])
      client_test_marker(client_id, count, load_gene(GENE_NAME(client_id), run_name))
      time.sleep(3)
      kwargs['count'] = count+1

      alg = Algorithm(run_name, recall=recall, num_clients=num_clients, **kwargs)
      gene = alg.create_gene()
      write_gene(gene, GENE_NAME(client_id), run_name)     # TODO: Naming
      kwargs.pop('fitness')   # TODO: Figure out more elegant solution
      self.run_client(**kwargs)   # client_name, gene_name included in kwargs
      return

    # Create base directory
    mkdir(self.run_name)

    # Start GA algorithm
    self.alg = Algorithm(self.run_name, **kwargs)

    # Run clients (generate id's)
    for i in range(num_clients):
      # TODO: Gene tied to client id, not necessarily
      self.run_client(client_id=i, gene_name=GENE_NAME(i), **kwargs)

  # Recall function
  def run_client(self, client_id: int, gene_name: str, **kwargs):
    ### Necessary kwargs for client run ###
    kwargs['gene_name'] = gene_name
    kwargs['client_id'] = client_id
    kwargs['run_name'] = self.run_name
    kwargs['call_type'] = "run_client"

    # Convert args/kwargs to bash
    bash_args = []
    for k,v in kwargs.items():
      bash_args.append(f"--{k}={v}")

    # Call client through terminal
    subprocess.Popen(["python", "GA.py"] + bash_args, shell=True)
    # bash_args = ' '.join(bash_args)
    # os.system("bash" + f" ./{CLIENT_RUNNER} " + bash_args)


class Algorithm():
  def __init__(self, run_name: str, num_genes: int = 10, recall: bool = False, **kwargs):
    self.run_name = run_name
    self.num_genes = num_genes
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = {}

    ### RECALL HANDLING ###
    if recall:
      fitness = kwargs.pop('fitness')   # *Always* pop fitness here
      fitness = float(fitness)

      # Load gene pool so new genes can be created
      for i in range(num_genes):
        gene = load_gene(GENE_NAME(i), run_name)
        if GENE_NAME(i) == GENE_NAME(kwargs['client_id']):  # Add new fitness
          self.pool[get_pool_key(gene['gene'])] = fitness
        else:
          self.pool[get_pool_key(gene['gene'])] = gene['fitness']
      print(f"POOL (RECALL): {self.pool}")
      return

    # Generate pool & files
    mkdir(self.pool_path)
    for i in range(num_genes):
      gene = self.create_gene()
      self.pool[get_pool_key(gene['gene'])] = -1
      write_gene(gene, GENE_NAME(i), run_name)
    print(f"POOL (INIT): {self.pool}")

  # TODO: Add kwargs
  def create_gene(self):
    # If pool uninitialized
    if len(self.pool) < self.num_genes:
      return {'gene': np.random.rand(10), 'name': "", 'fitness': -1}

    # If untested gene
    for gene_key, fitness in self.pool.items():
      if fitness < 0:
        return {'gene': get_gene(gene_key), 'fitness': fitness}

    # Create hybrid based on top 2 genes
    # print("HYBRID")
    ordered_genes = sorted(self.pool.items(), key=lambda x: x['fitness'], reverse=True)
    p1 = ordered_genes[0]['gene']
    p2 = ordered_genes[1]['gene']
    crossover_point = np.random.randint(1, len(p1) - 1)
    offspring_gene = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
    return offspring_gene


class Client():
  def __init__(self, run_name: str, client_id: int, gene_name: str, model, **kwargs):
    self.run_name = run_name
    self.client_id = client_id
    self.gene_name = gene_name
    self.model = model
    self.gene = load_gene(gene_name, run_name)
    print(f"CLIENT RECEIVED GENE: {self.gene}")

  def run(self, **kwargs):
    # Run model
    fitness = self.model.run(self.gene)

    # Write fitness (attached to gene)
    self.gene['fitness'] = fitness
    # print(f"CLIENT GENE: {self.gene}")
    write_gene(self.gene, gene_name, run_name)

    # Initiate callback
    self.callback(fitness, **kwargs)

  def callback(self, fitness: int, **kwargs):
    # Convert args/kwargs to bash
    kwargs['gene_name'] = self.gene_name
    kwargs['client_id'] = self.client_id
    kwargs['run_name'] = self.run_name
    kwargs['fitness'] = fitness
    kwargs['call_type'] = "server_callback"

    # Convert args/kwargs to bash
    bash_args = []
    for k, v in kwargs.items():
      bash_args.append(f"--{k}={v}")

    # Callback server through terminal
    subprocess.Popen(["python", "GA.py"] + bash_args, shell=True)
    # bash_args = ' '.join(bash_args)
    # os.system("bash" + f" ./{SERVER_CALLBACK} " + bash_args)


if __name__ == '__main__':
  # Parse unknown num. of arguments. *All strings*
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())
  call_type = all_args.pop('call_type')

  # Handle server calling client
  if call_type == "run_client":
    # Pop important params
    kwargs = all_args
    gene_name = kwargs.pop('gene_name')
    client_id = kwargs.pop('client_id')
    run_name = kwargs.pop('run_name')

    # Run client
    model = Model()   # TODO: Figure out how to pass models
    client = Client(run_name, client_id, gene_name, model, **kwargs)
    client.run(**kwargs)

  # Handle client callback to server
  elif call_type == "server_callback":
    # Pop important params
    kwargs = all_args
    run_name = kwargs.pop('run_name')
    server = Server(run_name, model_name='placeholder', num_clients=1, recall=True, **kwargs) # TODO: Client num

  else:
    print(f"error, improper call_type: {call_type}")
