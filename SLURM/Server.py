import subprocess
import json
from os.path import join as file_path
import portalocker
import sys
import argparse
import time
from pool_functions import write_gene, load_gene

# Constants for filesystem
POOL_DIR = "pool"
LOG_DIR = "logs"
LOCK_DIR = "locks"
POOL_LOCK_NAME = "POOL_LOCK.lock"


# Generate bash args from kwargs dict
def make_bash_args(**kwargs):
  bash_args = ["python3", "Server.py"]
  for key, val in kwargs.items():
    bash_args.append(f"--{key}={val}")
  return bash_args


class Server:
  def __init__(self, run_name: str, algorithm_path: str, algorithm_name: str, client_path: str, client_name: str,
               num_clients: int, iterations: int, call_type: str = 'init', **kwargs):

    # Load algorithm and client classes
    self.algorithm = getattr(__import__(algorithm_path, fromlist=[algorithm_name]), algorithm_name)
    self.client = getattr(__import__(client_path, fromlist=[client_name]), client_name)
    self.run_name = run_name
    self.algorithm_path = algorithm_path
    self.algorithm_name = algorithm_name
    self.client_path = client_path
    self.client_name = client_name
    self.num_clients = num_clients
    self.iterations = iterations

    if call_type == "init":

      # Generate initial 10 genes
      alg = self.algorithm(run_name=run_name, **kwargs)
      init_genes = []
      for i in range(num_clients):
        init_genes.append(alg.fetch_gene())

      # Call 1 client for each gene (and initialize count for iterations)
      count = 0
      bash_args = make_bash_args(run_name=run_name, algorithm_path=algorithm_path, algorithm_name=algorithm_name,
                                 client_path=client_path, client_name=client_name, num_clients=num_clients,
                                 iterations=iterations, call_type="run_client",
                                 count=count, **kwargs)
      for i, (g_name, _) in enumerate(init_genes):
        p = subprocess.Popen(bash_args + [f"--gene_name={g_name}"] + [f"--client_id={i}"])

    elif call_type == "run_client":
      self.run_client(**kwargs)

    elif call_type == "server_callback":
      self.server_callback(**kwargs)

    else:
      raise Exception(f"error, improper call_type: {call_type}")

  def run_client(self, **kwargs):
    # Run gene
    gene_name = kwargs['gene_name']
    clnt = self.client(self.run_name, gene_name)
    fitness = clnt.run()

    # Return fitness (by writing to files)
    gene_data = clnt.gene_data
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
    bash_args = make_bash_args(run_name=self.run_name, algorithm_path=self.algorithm_path, algorithm_name=self.algorithm_name,
                               client_path=self.client_path, client_name=self.client_name, num_clients=self.num_clients,
                               iterations=self.iterations, call_type="server_callback", **kwargs)
    p = subprocess.Popen(bash_args)

  def server_callback(self, **kwargs):
    count = int(kwargs.pop('count'))
    iterations = int(self.iterations)  # Comes back as str from bash
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
    bash_args = make_bash_args(run_name=self.run_name, algorithm_path=self.algorithm_path, algorithm_name=self.algorithm_name,
                               client_path=self.client_path, client_name=self.client_name, num_clients=self.num_clients,
                               iterations=iterations, call_type="run_client", gene_name=gene_name,
                               count=count, **kwargs)
    p = subprocess.Popen(bash_args)

  def write_logs(self, run_name: str, log_name: str, log_data: dict):

    # TODO: temporary solution, make this more general
    log_data['gene_data']['gene'] = log_data['gene_data']['gene'].tolist()

    log_path = file_path(run_name, LOG_DIR, log_name) + ".log"
    with open(log_path, 'a') as log_file:
      log_file.write(json.dumps(log_data) + "\n")


# Main function catches server-callbacks & runs clients
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())

  # Run server protocol with bash kwargs
  Server(**all_args)
