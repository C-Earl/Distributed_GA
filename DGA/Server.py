import copy
import os.path
import subprocess
import inspect
from os.path import join as file_path

import numpy as np
import portalocker
import sys
import argparse
import time
from DGA.pool_functions import write_gene_file, load_gene_file, POOL_DIR, LOG_DIR, ARGS_FOLDER, POOL_LOCK_NAME, write_log, \
  write_client_args_to_file, load_client_args_from_file, read_run_status, write_run_status, write_pool_log, \
  delete_gene_file, write_error_log
from DGA.Algorithm import Genetic_Algorithm_Base as Algorithm
from DGA.Client import Client


# https://stackoverflow.com/questions/17075071/is-there-a-python-method-to-access-all-non-private-and-non-builtin-attributes-of
# Helper function to get all public attributes of an object
def list_public_attributes(input_var):
  return [k for k, v in vars(input_var).items() if
          not (k.startswith('_') or callable(v))]


def dict_compare(d1, d2):
  for k, v in d1.items():
    if isinstance(v, dict):
      dict_compare(v, d2[k])
    elif isinstance(v, np.ndarray):
      if not np.array_equal(v, d2[k]):
        return False
    else:
      if v != d2[k]:
        return False
  return True


class Server:
  def __init__(self, run_name: str, algorithm: Algorithm | type, client: Client | type,
               num_parallel_processes: int, call_type: str = 'init',
               data_path: str = None, log_pool: int = -1, **kwargs):

    # self.algorithm = type(algorithm)      # Algorithm Class
    # self.client = type(client)            # Client Class
    self.run_name = run_name        # Name of run (used for folder name)
    self.num_parallel_processes = num_parallel_processes
    self.data_path = data_path      # Location of data folder (if needed, for async loading)
    self.server_file_path = os.path.abspath(__file__)   # Note: CWD not the same as DGA folder
    self.log_pool = log_pool        # Log pool-state every n genes (-1 for no logging)

    # Switch for handling client, server, or run initialization
    if call_type == "init":

      algorithm_type = type(algorithm)
      client_type = type(client)

      # Retrieve args (set by the user) passed to Client and Algorithm objects
      algorithm_args = list_public_attributes(algorithm)
      client_args = list_public_attributes(client)
      algorithm_args = {key: algorithm.__dict__[key] for key in algorithm_args}
      self.client_args = {key: client.__dict__[key] for key in client_args}

      # Define paths to client and algorithm files (used for loading in subprocess)
      self.algorithm_path = os.path.abspath(sys.modules[algorithm_type.__module__].__file__)
      self.client_path = os.path.abspath(sys.modules[client_type.__module__].__file__)
      self.algorithm_name = algorithm_type.__name__
      self.client_name = client_type.__name__

      self.init(algorithm_type, algorithm_args, **kwargs)

    elif call_type == "run_client":
      client_type = client    # Note: client will always be 'type' here, not object. 'run_client' called by Server not user

      # Set args passed to Client and Algorithm objects
      self.client_args = kwargs.pop('client_args')

      # Reload paths to client and algorithm files (used for loading in subprocess)
      self.algorithm_path = kwargs.pop('algorithm_path')
      self.client_path = kwargs.pop('client_path')
      self.algorithm_name = kwargs.pop('algorithm_name')
      self.client_name = kwargs.pop('client_name')
      self.run_client(client_type, **kwargs)

    elif call_type == "server_callback":
      algorithm_type = algorithm    # Note: alg will always be 'type' here, not object. 'server_callback' called by Server not user

      # Set args passed to Client and Algorithm objects
      self.client_args = kwargs.pop('client_args')

      # Reload paths to client and algorithm files (used for loading in subprocess)
      self.algorithm_path = kwargs.pop('algorithm_path')
      self.client_path = kwargs.pop('client_path')
      self.algorithm_name = kwargs.pop('algorithm_name')
      self.client_name = kwargs.pop('client_name')
      self.server_callback(algorithm_type, **kwargs)

    else:
      raise Exception(f"error, improper call_type: {call_type}")

  # Initialization procedure:
  # 1. Create directories for run
  # 2. Create run status file. Status file represents state of genetic algorithm (sync. between all clients)
  # 3. Generate initial genes (and save updated status)
  # 4. Call clients to run initial genes
  def init(self, algorithm_type: type, algorithm_args: dict, **kwargs):

    # Make directory if needed
    # TODO: This prevents continuing runs.
    os.makedirs(file_path(self.run_name, POOL_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, LOG_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, ARGS_FOLDER), exist_ok=True)

    # Create run status file
    write_run_status(self.run_name, algorithm_args)

    # Generate initial genes
    alg = algorithm_type(run_name=self.run_name, **algorithm_args)
    original_pool = copy.deepcopy(alg.pool)
    init_genes = []
    for i in range(self.num_parallel_processes):
      init_genes.append(alg.fetch_gene()[0])    # Don't need status on init, just gene

    # Update pool files
    final_pool = alg.pool
    self.update_pool(original_pool, final_pool)

    # Update status
    self.update_run_status(alg, algorithm_args.keys())

    # Call clients to run initial genes
    for i, g_name in enumerate(init_genes):
      self.make_call(i, g_name, "run_client", **kwargs)

  def run_client(self, client_type: type, **kwargs):
    # Setup client
    gene_name = kwargs['gene_name']
    gene_data = load_gene_file(self.run_name, gene_name)
    client = client_type(**self.client_args)

    # Load data using file locks (presumably training data)
    data_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    with portalocker.Lock(data_lock_path, timeout=100) as _:
      client.load_data()

    # Test gene
    fitness = client.run(gene_data['gene'], **kwargs)

    # Return fitness (by writing to files)
    gene_data['fitness'] = fitness
    gene_data['test_state'] = 'tested'
    pool_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    with portalocker.Lock(pool_lock_path, timeout=100) as _:
      write_gene_file(self.run_name, gene_name, gene_data)
      write_log(self.run_name, kwargs['client_id'], client.logger(fitness))

    # Callback server
    self.client_args = {key: client.__dict__[key] for key in self.client_args}    # Update args
    self.make_call(call_type="server_callback", **kwargs)   # Other args contained in kwargs

  def server_callback(self, algorithm_type: type, **kwargs):

    # Lock pool during gene creation
    pool_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    while True:
      with portalocker.Lock(pool_lock_path, timeout=100) as _:

        # Setup algorithm & get copy of original pool state
        algorithm_args = read_run_status(self.run_name)
        alg = algorithm_type(run_name=self.run_name, **algorithm_args)
        orginal_pool = copy.deepcopy(alg.pool)

        # Check if run is complete
        if alg.end_condition():
          sys.exit()

        # Fetch next gene for testing
        gene_name, success = alg.fetch_gene()

        # Update status & break if fetch was success, otherwise loops
        if success:

          # Update pool files
          final_pool = alg.pool
          self.update_pool(orginal_pool, final_pool)

          # Update pool log
          if self.log_pool != -1 and alg.current_iter % self.log_pool == 0:
            write_pool_log(self.run_name, alg.pool)

          # Update run status
          self.update_run_status(alg, algorithm_args.keys())

          break
        else:
          time.sleep(1)

    # Remove old gene_name from args, and send new gene to client
    kwargs.pop('gene_name')
    self.make_call(call_type="run_client", gene_name=gene_name, **kwargs)

  def update_pool(self, original_pool: dict, new_pool: dict):
    # Injective check from new_pool to original_pool
    for gene_name in new_pool.keys():
      if gene_name not in original_pool.keys():     # If it's a newly added gene
        new_gene_data = new_pool[gene_name]
        write_gene_file(self.run_name, gene_name, new_gene_data)
      # else:                                         # This didn't work due to EOF error (pickle)
      #   if dict_compare(new_pool[gene_name], original_pool[gene_name]):   # ...and it's changed
      #     new_gene_data = new_pool[gene_name]
      #     write_gene_file(self.run_name, gene_name, new_gene_data)

    # Injective check from original_pool to new_pool
    for gene_name in original_pool.keys():
      if gene_name not in new_pool.keys():          # If it's a removed gene
        delete_gene_file(self.run_name, gene_name)

  # Save important args to file and run next phase (callback or run_client)
  # Saves here are per-client, so that each client can run independently
  def make_call(self,
                client_id: int,
                gene_name: str,
                call_type: str,
                **kwargs):

    write_client_args_to_file(client_id=client_id,
                              gene_name=gene_name,
                              call_type=call_type,  # callback or run_client
                              run_name=self.run_name,
                              algorithm_path=self.algorithm_path,
                              algorithm_name=self.algorithm_name,
                              client_path=self.client_path,
                              client_name=self.client_name,
                              client_args=self.client_args,
                              num_parallel_processes=self.num_parallel_processes,
                              data_path=self.data_path,
                              log_pool=self.log_pool,
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

  # Update status file with new args
  def update_run_status(self, algorithm: Algorithm, args: list):
    algorithm_args = {key: algorithm.__dict__[key] for key in args}  # Update args
    write_run_status(self.run_name, algorithm_args)

# Main function catches server-callbacks & runs clients
if __name__ == '__main__':
  parser_ = argparse.ArgumentParser()
  parser_.add_argument('--client_id', type=int)
  parser_.add_argument('--run_name', type=str)
  args_ = parser_.parse_args()

  # Load args from file
  all_args = load_client_args_from_file(args_.client_id, args_.run_name)

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
  try:
    Server(**all_args)
  except Exception as e:
    write_error_log(all_args['run_name'], all_args)
    raise e