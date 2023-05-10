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
      client_id = kwargs['client_id']

      # TODO: Testing code
      count = int(kwargs['count'])
      client_test_marker(client_id, count, load_gene(GENE_NAME(client_id), run_name))
      time.sleep(3)
      kwargs['count'] = count+1

      alg = Algorithm(run_name, recall=recall, num_clients=num_clients, **kwargs)
      gene = alg.create_gene()
      write_gene(gene, GENE_NAME(client_id), run_name)     # TODO: Naming
      self.run_client(client_id, **kwargs)   # gene_name included in kwargs
      return

    # Create base directory
    mkdir(self.run_name)

    # Start GA algorithm
    self.alg = Algorithm(self.run_name, **kwargs)

    # Run clients (generate id's)
    for i in range(num_clients):
      # TODO: Gene tied to client id, not necessarily
      self.run_client(id=i, gene_name=GENE_NAME(i), **kwargs)

  # Recall function
  def run_client(self, id: int, gene_name: str, **kwargs):
    ### Necessary kwargs for client run ###
    kwargs['gene_name'] = gene_name
    kwargs['client_id'] = id
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
    self.pool_path = file_path(self.run_name, POOL_DIR)
    self.pool = {}

    ### RECALL HANDLING ###
    if recall:
      # Load gene pool so new genes can be created
      for i in range(num_genes):
        gene = load_gene(GENE_NAME(i), run_name)
        self.pool[i] = gene
      return

    # Generate pool & files
    mkdir(self.pool_path)
    for i in range(num_genes):
      gene = self.create_gene()
      self.pool[i] = gene
      write_gene(gene, GENE_NAME(i), run_name)

  # TODO: Add kwargs
  def create_gene(self):
    return {'gene' : np.random.rand(10), 'fitness' : -1}


class Client():
  def __init__(self, run_name: str, client_id: int, gene_name: str, model, **kwargs):
    self.run_name = run_name
    self.client_id = client_id
    self.gene_name = gene_name
    self.model = model
    self.gene = load_gene(gene_name, run_name)

  def run(self, **kwargs):
    # Run model
    fitness = self.model.run(self.gene)

    # Write fitness (attached to gene)
    self.gene['fitness'] = fitness
    write_gene(self.gene, gene_name, run_name)

    # Initiate callback
    self.callback(fitness, **kwargs)

  def callback(self, fitness: int, **kwargs):

    # Convert args/kwargs to bash
    bash_args = []
    bash_args.append(f"--gene_name={self.gene_name}")      # Send Server gene tested
    bash_args.append(f"--client_id={self.client_id}")             # and ID
    bash_args.append(f"--run_name={self.run_name}")   # and what run its in
    bash_args.append(f"--fitness={fitness}")
    bash_args.append(f"--call_type=server_callback")
    for k, v in kwargs.items():
      bash_args.append(f"--{k}={v}")

    # Callback server through terminal
    out = subprocess.Popen(["python", "GA.py"] + bash_args, shell=True)
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

  # Define pool location

  # Handle server calling client
  if call_type == "run_client":
    kwargs = all_args
    gene_name = kwargs.pop('gene_name')
    client_id = kwargs.pop('client_id')
    run_name = kwargs.pop('run_name')
    model = Model()   # TODO
    client = Client(run_name, client_id, gene_name, model, **kwargs)
    client.run(**kwargs)

  # Handle client callback to server
  elif call_type == "server_callback":
    kwargs = all_args
    run_name = kwargs.pop('run_name')
    fitness = kwargs.pop('fitness')

    # TODO: Algorithm specific
    kwargs['num_genes'] = int(kwargs['num_genes'])

    server = Server(run_name, model_name='placeholder', num_clients=1, recall=True, **kwargs) # TODO: Client num

  else:
    print(f"error, improper call_type: {call_type}")
