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
TEST_DIR = "test_dir"
POOL_LOCK_NAME = "POOL_LOCK.lock"

class Server():
  def __init__(self, algorithm_path: str, algorithm_name: str, client_path: str, client_name: str, **kwargs):

    CALL_TYPE = kwargs.pop('call_type')
    RUN_NAME = "test_dir"
    GENE_SHAPE = 10  # TODO: SET THESE TO KWARGS
    MUTATION_RATE = 0.2
    NUM_GENES = 10

    # Load algorithm and client classes
    algorithm = getattr(__import__(algorithm_path, fromlist=[algorithm_name]), algorithm_name)
    client = getattr(__import__(client_path, fromlist=[client_name]), client_name)

    if CALL_TYPE == "init":

      # Generate initial 10 genes
      alg = algorithm(RUN_NAME, GENE_SHAPE, MUTATION_RATE, num_genes=NUM_GENES)
      init_genes = []
      for i in range(NUM_GENES):
        init_genes.append(alg.fetch_gene())

      # Call 1 client for each gene
      for g_name, _ in init_genes:
        p = subprocess.Popen(["python3", "Server.py", "--call_type=run_client", f"--gene_name={g_name}",
                              f"--algorithm_path={algorithm_path}", f"--algorithm_name={algorithm_name}",
                              f"--client_path={client_path}", f"--client_name={client_name}",
                              f"--count={kwargs['count']}"])

    elif CALL_TYPE == "run_client":

      # Run gene
      gene_name = kwargs['gene_name']
      clnt = client(RUN_NAME, gene_name)
      fitness = clnt.run()

      # Return fitness (by writing to files)
      gene_data = clnt.gene_data
      gene_data['fitness'] = fitness
      gene_data['status'] = 'tested'
      pool_lock_path = file_path(RUN_NAME, POOL_LOCK_NAME)
      with portalocker.Lock(pool_lock_path, timeout=100) as _:
        write_gene(gene_data, gene_name, RUN_NAME)

      count = int(kwargs['count'])
      p = subprocess.Popen(["python3", "Server.py", "--call_type=server_callback",
                            f"--algorithm_path={algorithm_path}", f"--algorithm_name={algorithm_name}",
                            f"--client_path={client_path}", f"--client_name={client_name}",
                            f"--count={count}"])

    elif CALL_TYPE == "server_callback":
      count = int(kwargs['count'])
      count += 1
      if count >= 50:
        sys.exit()

      # Lock pool during gene creation
      pool_lock_path = file_path(RUN_NAME, POOL_LOCK_NAME)
      while True:
        with portalocker.Lock(pool_lock_path, timeout=100) as _:

          # Init alg (loads gene pool)
          alg = algorithm(RUN_NAME, GENE_SHAPE, MUTATION_RATE, NUM_GENES)

          # Fetch next gene for testing
          gene_name, success = alg.fetch_gene()

        # Break if fetch was success, otherwise loops
        if success:
          break
        else:
          time.sleep(1)

      p = subprocess.Popen(["python3", "Server.py", "--call_type=run_client", f"--gene_name={gene_name}",
                            f"--algorithm_path={algorithm_path}", f"--algorithm_name={algorithm_name}",
                            f"--client_path={client_path}", f"--client_name={client_name}",
                            f"--count={count}"])

    else:
      raise Exception(f"error, improper call_type: {CALL_TYPE}")


# Main function catches server-callbacks & runs clients
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())

  # Run server protocol with bash kwargs
  Server(**all_args)
