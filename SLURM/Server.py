import pickle
import subprocess
from os.path import join as file_path
import portalocker
import sys
import argparse
import time
from pool_functions import write_gene

POOL_DIR = "pool"
LOCK_DIR = "locks"
POOL_LOCK_NAME = "POOL_LOCK.lock"


def make_bash_args(**kwargs):
  bash_args = ["python3", "Server.py"]
  for key, val in kwargs.items():
    bash_args.append(f"--{key}={val}")
  return bash_args


class Server():
  def __init__(self, run_name: str, algorithm_path: str, algorithm_name: str, client_path: str, client_name: str,
               num_clients: int, call_type: str = 'init', **kwargs):

    # Load algorithm and client classes
    algorithm = getattr(__import__(algorithm_path, fromlist=[algorithm_name]), algorithm_name)
    client = getattr(__import__(client_path, fromlist=[client_name]), client_name)

    if call_type == "init":

      # Generate initial 10 genes
      alg = algorithm(run_name=run_name, **kwargs)
      init_genes = []
      for i in range(num_clients):
        init_genes.append(alg.fetch_gene())

      # Call 1 client for each gene
      bash_args = make_bash_args(run_name=run_name, algorithm_path=algorithm_path, algorithm_name=algorithm_name,
                                 client_path=client_path, client_name=client_name, num_clients=num_clients,
                                 call_type="run_client", **kwargs)
      for g_name, _ in init_genes:
        p = subprocess.Popen(bash_args + [f"--gene_name={g_name}"])

    elif call_type == "run_client":

      # Run gene
      gene_name = kwargs['gene_name']
      clnt = client(run_name, gene_name)
      fitness = clnt.run()

      # Return fitness (by writing to files)
      gene_data = clnt.gene_data
      gene_data['fitness'] = fitness
      gene_data['status'] = 'tested'
      pool_lock_path = file_path(run_name, POOL_LOCK_NAME)
      with portalocker.Lock(pool_lock_path, timeout=100) as _:
        write_gene(gene_data, gene_name, run_name)

      # Callback server
      # kwargs.pop('count')
      bash_args = make_bash_args(run_name=run_name, algorithm_path=algorithm_path, algorithm_name=algorithm_name,
                                 client_path=client_path, client_name=client_name, num_clients=num_clients,
                                 call_type="server_callback", **kwargs)
      p = subprocess.Popen(bash_args)

    elif call_type == "server_callback":
      count = int(kwargs.pop('count'))
      count += 1
      if count >= 50:
        sys.exit()

      # Lock pool during gene creation
      pool_lock_path = file_path(run_name, POOL_LOCK_NAME)
      while True:
        with portalocker.Lock(pool_lock_path, timeout=100) as _:

          # Init alg (loads gene pool)
          alg = algorithm(run_name=run_name, **kwargs)

          # Fetch next gene for testing
          gene_name, success = alg.fetch_gene()

        # Break if fetch was success, otherwise loops
        if success:
          break
        else:
          time.sleep(1)

      # Remove old gene_name from args, and send new gene to client
      kwargs.pop('gene_name')
      bash_args = make_bash_args(run_name=run_name, algorithm_path=algorithm_path, algorithm_name=algorithm_name,
                                 client_path=client_path, client_name=client_name, num_clients=num_clients,
                                 count=count, call_type="run_client", gene_name=gene_name, **kwargs)
      p = subprocess.Popen(bash_args)

    else:
      raise Exception(f"error, improper call_type: {call_type}")


# Main function catches server-callbacks & runs clients
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())

  # Run server protocol with bash kwargs
  Server(**all_args)
