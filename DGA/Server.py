import os.path
import subprocess
import inspect
from os.path import join as file_path
import portalocker
import sys
import argparse
import time
from DGA.pool_functions import write_gene, load_gene, POOL_DIR, LOG_DIR, ARGS_FOLDER, POOL_LOCK_NAME, write_log, \
  write_args_to_file, load_args_from_file, read_run_status, write_run_status
from DGA.Algorithm import Genetic_Algorithm_Base as Algorithm
from DGA.Client import Client


class Server:
  def __init__(self, run_name: str, algorithm: Algorithm, client: Client,
               num_parallel_processes: int, call_type: str = 'init',
               data_path: str = None, **kwargs):

    self.algorithm = type(algorithm)      # Algorithm Class
    self.client = type(client)            # Client Class
    self.run_name = run_name        # Name of run (used for folder name)
    self.num_parallel_processes = num_parallel_processes
    self.data_path = data_path      # Location of data folder (if needed, for async loading)
    self.server_file_path = os.path.abspath(__file__)   # Note: CWD not the same as DGA folder

    # Switch for handling client, server, or run initialization
    if call_type == "init":

      # Retrieve args passed to Client and Algorithm objects
      algorithm_args = inspect.getfullargspec(algorithm.__init__).args[1:]
      client_args = inspect.getfullargspec(client.__init__).args[1:]
      self.algorithm_args = {key: algorithm.__dict__[key] for key in algorithm_args}
      self.client_args = {key: client.__dict__[key] for key, val in client_args}

      # current_iter not a GA argument, so manually inserted
      self.algorithm_args.update({'current_iter': 1})

      # Define paths to client and algorithm files (used for loading in subprocess)
      self.algorithm_path = os.path.abspath(sys.modules[self.algorithm.__module__].__file__)
      self.client_path = os.path.abspath(sys.modules[self.client.__module__].__file__)
      self.algorithm_name = self.algorithm.__name__
      self.client_name = self.client.__name__

      self.init(**kwargs)

    elif call_type == "run_client":

      # Set args passed to Client and Algorithm objects
      self.algorithm_args = kwargs.pop('algorithm_args')
      self.client_args = kwargs.pop('client_args')

      # Reload paths to client and algorithm files (used for loading in subprocess)
      self.algorithm_path = kwargs.pop('algorithm_path')
      self.client_path = kwargs.pop('client_path')
      self.algorithm_name = kwargs.pop('algorithm_name')
      self.client_name = kwargs.pop('client_name')
      self.run_client(**kwargs)

    elif call_type == "server_callback":

      # Set args passed to Client and Algorithm objects
      self.algorithm_args = kwargs.pop('algorithm_args')
      self.client_args = kwargs.pop('client_args')

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
    os.makedirs(file_path(self.run_name, POOL_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, LOG_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, ARGS_FOLDER), exist_ok=True)

    # Generate initial genes
    alg = self.algorithm(run_name=self.run_name, init_run=True, **self.algorithm_args)
    init_genes = []
    for i in range(self.num_parallel_processes):
      init_genes.append(alg.fetch_gene()[0])    # Don't need status, just gene

    # Call clients to run initial genes
    for i, g_name in enumerate(init_genes):
      self.make_call(i, g_name, "run_client", **kwargs)

  def run_client(self, **kwargs):
    # Setup client
    gene_name = kwargs.get('gene_name', None)
    gene_data = load_gene(gene_name, self.run_name)
    clnt = self.client(**self.client_args)

    # Load data using file locks (presumably training data)
    data_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    with portalocker.Lock(data_lock_path, timeout=100) as _:
      clnt.load_data()

    # Test gene
    fitness = clnt.run(gene_data['gene'], **kwargs)

    # Return fitness (by writing to files)
    gene_data['fitness'] = fitness
    gene_data['test_state'] = 'tested'
    pool_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    with portalocker.Lock(pool_lock_path, timeout=100) as _:
      write_gene(gene_data, gene_name, self.run_name)
      write_log(self.run_name, kwargs['client_id'], clnt.logger(fitness))

    # Callback server
    self.client_args = {key: clnt.__dict__[key] for key in self.client_args}    # Update args
    self.make_call(call_type="server_callback", **kwargs)   # Other args contained in kwargs

  def server_callback(self, **kwargs):

    # Lock pool during gene creation
    pool_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    while True:
      with portalocker.Lock(pool_lock_path, timeout=100) as _:

        # Setup algorithm
        alg = self.algorithm(run_name=self.run_name, **self.algorithm_args)

        # Check if run is complete
        if alg.end_condition():
          sys.exit()

        # Fetch next gene for testing
        gene_name, success = alg.fetch_gene()

      # Break if fetch was success, otherwise loops
      if success:
        break
      else:
        time.sleep(1)

    # Remove old gene_name from args, and send new gene to client
    kwargs.pop('gene_name')
    self.algorithm_args = {key: alg.__dict__[key] for key in self.algorithm_args}    # Update args
    self.make_call(call_type="run_client", gene_name=gene_name, **kwargs)

  # Save important args to file and run next phase (callback or run_client)
  def make_call(self, client_id: int, gene_name: str, call_type: str, **kwargs):
    write_args_to_file(client_id=client_id,
                       gene_name=gene_name,
                       call_type=call_type,   # callback or run_client
                       run_name=self.run_name,
                       algorithm_path=self.algorithm_path,
                       algorithm_name=self.algorithm_name,
                       client_path=self.client_path,
                       client_name=self.client_name,
                       # algorithm_args=self.algorithm_args,
                       client_args=self.client_args,
                       num_parallel_processes=self.num_parallel_processes,
                       data_path=self.data_path,
                       **kwargs)
    write_run_status(self.run_name, self.algorithm_args)

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
  algorithm_args_ = read_run_status(args_.run_name) # Arguments required for Algorithm obj
  all_args['algorithm_args'] = algorithm_args_
  client_args_ = all_args['client_args']            # Arguments required for Client obj
  all_args['algorithm'] = algorithm_(**algorithm_args_)
  all_args['client'] = client_(**client_args_)

  # Run server protocol with bash kwargs
  Server(**all_args)
