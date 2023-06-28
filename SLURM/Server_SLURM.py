import pickle
from os import system as cmd
from Server import Server
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
ARGS_FOLDER = "run_args"
POOL_LOCK_NAME = "POOL_LOCK.lock"


# Generate bash args from kwargs dict
# def make_bash_args(**kwargs):
#   bash_args = ["python3", "Server.py"]
#   for key, val in kwargs.items():
#     bash_args.append(f"--{key}={val}")
#   return bash_args

def write_args_to_file(client_id: int, **kwargs):
  args_path = file_path(kwargs['run_name'], ARGS_FOLDER, f"client{client_id}_args.txt")
  kwargs['client_id'] = client_id
  pickle.dump(kwargs, open(args_path, 'wb'))


def load_args_from_file(client_id: int, run_name: str):
  args_path = file_path(run_name, ARGS_FOLDER, f"client{client_id}_args.txt")
  return pickle.load(open(args_path, 'rb'))


# sbatch_script should be sh file preparing environment for model testing; user's responsibility
class SLURM_Server(Server):
  def __init__(self, run_name: str, algorithm_path: str, algorithm_name: str, client_path: str, client_name: str,
               num_clients: int, iterations: int, sbatch_script: str, call_type: str = 'init', **kwargs):
    self.sbatch_script = sbatch_script
    super().__init__(run_name, algorithm_path, algorithm_name, client_path, client_name, num_clients, iterations, call_type, **kwargs)

  def init(self, **kwargs):
    # Generate initial 10 genes
    alg = self.algorithm(run_name=self.run_name, **kwargs)
    init_genes = []
    for i in range(self.num_clients):
      init_genes.append(alg.fetch_gene())

    # Call 1 client for each gene (and initialize count for iterations)
    count = 0
    for i, (g_name, _) in enumerate(init_genes):
      # p = subprocess.Popen(bash_args + [f"--gene_name={g_name}"] + [f"--client_id={i}"])
      # cmd(f"sbatch {self.sbatch_script} {' '.join(bash_args)} --gene_name={g_name} --client_id={i}")
      write_args_to_file(run_name=self.run_name, algorithm_path=self.algorithm_path, algorithm_name=self.algorithm_name,
                         client_path=self.client_path, client_name=self.client_name, num_clients=self.num_clients,
                         iterations=self.iterations, call_type="run_client", count=count,
                         gene_name=g_name, client_id=i, **kwargs)
      cmd(f"sbatch {self.sbatch_script} --client_id={i} --run_name={self.run_name}")

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

    self.server_callback(**kwargs)

  # "Callback" server (not real server, done on local machine)
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

    # Queue next node to test gene
    kwargs.pop('gene_name')
    bash_args = make_bash_args(run_name=self.run_name, algorithm_path=self.algorithm_path, algorithm_name=self.algorithm_name,
                               client_path=self.client_path, client_name=self.client_name, num_clients=self.num_clients,
                               iterations=iterations, call_type="run_client", gene_name=gene_name,
                               count=count, **kwargs)
    cmd(f"sbatch {self.sbatch_script} {' '.join(bash_args)}")


# Main function catches calls from run_server.sh
# Note: Args (aside from client_id and run_name) loaded via file, NOT bash args
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--client_id', type=int)
  parser.add_argument('--run_name', type=str)
  args = parser.parse_args()

  # Load args from file
  all_args = load_args_from_file(args.client_id, args.run_name)

  # Run server protocol with bash kwargs
  SLURM_Server(**all_args)
