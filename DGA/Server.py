import os.path
import subprocess
import json
import pickle
from os.path import join as file_path
import portalocker
import sys
import argparse
import time
from DGA.pool_functions import write_gene, load_gene
from DGA.Algorithm import Algorithm
from DGA.Client import Client
from typing import Type

# Constants for filesystem
POOL_DIR = "pool"
LOG_DIR = "logs"
ARGS_FOLDER = "run_args"
POOL_LOCK_NAME = "POOL_LOCK.lock"
DATA_LOCK_NAME = "DATA_LOCK.lock"

def write_args_to_file(client_id: int, **kwargs):
  args_path = file_path(kwargs['run_name'], ARGS_FOLDER, f"client{client_id}_args.pkl")
  kwargs['client_id'] = client_id
  pickle.dump(kwargs, open(args_path, 'wb'))


def load_args_from_file(client_id: int, run_name: str):
  args_path = file_path(run_name, ARGS_FOLDER, f"client{client_id}_args.pkl")
  return pickle.load(open(args_path, 'rb'))


class Server:
  def __init__(self, run_name: str, algorithm: Type[Algorithm], client: Type[Client],
               num_parallel_processes: int, iterations: int, call_type: str = 'init',
               data_path: str = None, **kwargs):

    # Load algorithm and client classes
    self.algorithm = algorithm      # Algorithm Class
    self.client = client            # Client Class
    self.run_name = run_name
    self.num_parallel_processes = num_parallel_processes
    self.iterations = iterations    # iterations per subprocess
    self.data_path = data_path  # Location of data folder (if needed, for async loading)
    self.server_file_path = os.path.abspath(__file__)   # Note: CWD not the same as DGA folder

    # Switch for handling client, server, or run initialization
    if call_type == "init":

      # Define paths to client and algorithm files (used for loading in subprocess)
      self.algorithm_path = os.path.abspath(sys.modules[self.algorithm.__module__].__file__)
      self.client_path = os.path.abspath(sys.modules[self.client.__module__].__file__)
      self.algorithm_name = self.algorithm.__name__
      self.client_name = self.client.__name__
      self.init(**kwargs)

    elif call_type == "run_client":

      # Reload paths to client and algorithm files (used for loading in subprocess)
      self.algorithm_path = kwargs.pop('algorithm_path')
      self.client_path = kwargs.pop('client_path')
      self.algorithm_name = kwargs.pop('algorithm_name')
      self.client_name = kwargs.pop('client_name')
      self.run_client(**kwargs)

    elif call_type == "server_callback":
      # Reload paths to client and algorithm files (used for loading in subprocess)
      self.algorithm_path = kwargs.pop('algorithm_path')
      self.client_path = kwargs.pop('client_path')
      self.algorithm_name = kwargs.pop('algorithm_name')
      self.client_name = kwargs.pop('client_name')
      self.server_callback(**kwargs)

    else:
      raise Exception(f"error, improper call_type: {call_type}")

  def init(self, **kwargs):

    # Make directory if needed
    # Note: CWD will be at where user-written script is
    os.makedirs(file_path(self.run_name, POOL_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, LOG_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, ARGS_FOLDER), exist_ok=True)

    # Generate initial 10 genes
    alg = self.algorithm(run_name=self.run_name, **kwargs)
    init_genes = []
    for i in range(self.num_parallel_processes):
      init_genes.append(alg.fetch_gene())

    # Call 1 client for each gene (and initialize count for iterations)
    count = 0
    for i, (g_name, _) in enumerate(init_genes):
      self.make_call(i, g_name, "run_client", count, **kwargs)

  def run_client(self, **kwargs):
    # Setup client
    gene_name = kwargs['gene_name']
    gene_data = load_gene(gene_name, self.run_name)  # Note: Read should be safe as long as only 1 client runs gene
    clnt = self.client(self.run_name, gene_name)

    # Load data
    data_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    with portalocker.Lock(data_lock_path, timeout=100) as _:
      clnt.load_data()

    # Test gene
    fitness = clnt.run(gene_data['gene'], **kwargs)

    # Return fitness (by writing to files)
    gene_data['fitness'] = fitness
    gene_data['status'] = 'tested'
    pool_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    with portalocker.Lock(pool_lock_path, timeout=100) as _:
      write_gene(gene_data, gene_name, self.run_name)

      # Write gene to logs
      timestamp = time.strftime('%H:%M:%S', time.localtime())
      log_data = {'timestamp': timestamp, 'gene_name': gene_name, 'gene_data': gene_data}
      self.write_logs(self.run_name, kwargs['client_id'], log_data)  # Separate logs by client_id

    # Callback server
    self.make_call(call_type="server_callback", **kwargs)   # Other args contained in kwargs

  def server_callback(self, **kwargs):
    count = kwargs.pop('count')
    iterations = self.iterations
    count += 1
    if count >= iterations:
      sys.exit()

    # Lock pool during gene creation
    pool_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    while True:
      with portalocker.Lock(pool_lock_path, timeout=100) as _:

        # Init alg (loads gene pool)
        alg = self.algorithm(run_name=self.run_name, **kwargs)

        # Fetch next gene for testing
        gene_name, success = alg.fetch_gene()

      # Break if fetch was success, otherwise loops
      if success:
        break
      else:
        time.sleep(1)

    # Remove old gene_name from args, and send new gene to client
    kwargs.pop('gene_name')
    self.make_call(call_type="run_client", gene_name=gene_name, count=count, **kwargs)

  def write_logs(self, run_name: str, log_name: int, log_data: dict):

    # Convert to proper type for logging
    import numpy as np
    if isinstance(log_data['gene_data']['gene'], dict):
      for key, val in log_data['gene_data']['gene'].items():
        if isinstance(val, np.ndarray):
          log_data['gene_data']['gene'][key] = val.tolist()
        else:
          raise Exception(f"Unsupported gene type for logging: {type(val)}")
    elif isinstance(log_data['gene_data']['gene'], np.ndarray):
      log_data['gene_data']['gene'] = log_data['gene_data']['gene'].tolist()
    else:
      raise Exception(f"Unsupported gene type for logging: {log_data['gene_data']['gene']}")

    log_path = file_path(run_name, LOG_DIR, str(log_name)) + ".log"
    with open(log_path, 'a') as log_file:
      log_file.write(json.dumps(log_data) + "\n")

  # Save client info to file and run next phase (callback or run_client)
  def make_call(self, client_id: int, gene_name: str, call_type: str, count: int, **kwargs):
    write_args_to_file(client_id=client_id,
                       gene_name=gene_name,
                       call_type=call_type,   # callback or run_client
                       count=count,           # current iteration
                       run_name=self.run_name,
                       algorithm_path=self.algorithm_path,
                       algorithm_name=self.algorithm_name,
                       client_path=self.client_path,
                       client_name=self.client_name,
                       num_parallel_processes=self.num_parallel_processes,
                       iterations=self.iterations,
                       data_path=self.data_path,
                       **kwargs)

    # Run command according to OS
    # TODO: SOMEONE DO THIS FOR MAC PLEASE
    if sys.platform == "linux":
      p = subprocess.Popen(["python3", self.server_file_path, f"--run_name={self.run_name}", f"--client_id={client_id}"])
    elif sys.platform == "win32":
      p = subprocess.Popen(["python", self.server_file_path, f"--run_name={self.run_name}", f"--client_id={client_id}"],
                         shell=True)
    elif sys.platform == "darwin":
      pass    # MAC HANDLING


# Main function catches server-callbacks & runs clients
if __name__ == '__main__':
  parser_ = argparse.ArgumentParser()
  parser_.add_argument('--client_id', type=int)
  parser_.add_argument('--run_name', type=str)
  args_ = parser_.parse_args()

  # Load args from file
  all_args = load_args_from_file(args_.client_id, args_.run_name)

  # Establish location of Algorithm and Client classes & add them to python path
  algorithm_path_ = all_args['algorithm_path']
  algorithm_name_ = all_args['algorithm_name']
  client_path_ = all_args['client_path']
  client_name_ = all_args['client_name']
  server_path_ = os.path.abspath(__file__)  # Get absolute path to current location on machine
  base_path_ = '/'.join(server_path_.split('/')[0:-2])    # Get path to "./Distributed_GA" ie. base folder
  alg_module_path_ = file_path(base_path_, '/'.join(algorithm_path_.split('/')[0:-1]))
  client_module_path_ = file_path(base_path_, '/'.join(client_path_.split('/')[0:-1]))
  sys.path.append(alg_module_path_)
  sys.path.append(client_module_path_)

  # Create Algorithm and Client objects
  alg_module_name = algorithm_path_.split('/')[-1][:-3]
  client_module_name = client_path_.split('/')[-1][:-3]
  algorithm_ = getattr(__import__(alg_module_name, fromlist=[alg_module_name]), algorithm_name_)
  client_ = getattr(__import__(client_module_name, fromlist=[client_module_name]), client_name_)
  all_args['algorithm'] = algorithm_
  all_args['client'] = client_

  # Run server protocol with bash kwargs
  Server(**all_args)
