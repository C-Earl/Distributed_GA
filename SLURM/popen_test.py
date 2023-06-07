import subprocess
import pickle
from os.path import join as file_path
import portalocker
import sys
import argparse
import time
import numpy as np
import os
from Algorithm import Algorithm, write_gene
from Client import Client

POOL_DIR = "pool"
LOCK_DIR = "locks"
TEST_DIR = "test_dir"
TEST_GENE_NAME = "test_gene"
CLIENT_RUNNER = "run_client.sh"
SERVER_CALLBACK = "run_server.sh"


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())
  call_type = all_args.pop('call_type')

  RUN_NAME = "test_dir"
  POOL_LOCK_NAME = "POOL_LOCK.lock"
  GENE_SHAPE = 10
  MUTATION_RATE = 0.2
  NUM_GENES = 10

  if call_type == "init":
    alg = Algorithm(RUN_NAME, GENE_SHAPE, MUTATION_RATE, num_genes=NUM_GENES)
    init_genes = []
    for i in range(NUM_GENES):  # Generate initial 10 genes
      init_genes.append(alg.fetch_gene())

    for g_name, _ in init_genes:  # Call 1 client for each gene
      p = subprocess.Popen(["python3", "popen_test.py", "--call_type=run_client", f"--gene_name={g_name}",
                            f"--count={all_args['count']}"])

  elif call_type == "run_client":
    # time.sleep(np.random.rand())

    # Run gene
    gene_name = all_args['gene_name']
    client = Client(RUN_NAME, gene_name)
    fitness = client.run()

    # Return fitness (by writing to files)
    gene_data = client.gene_data
    gene_data['fitness'] = fitness
    gene_data['status'] = 'tested'
    pool_lock_path = file_path(RUN_NAME, POOL_LOCK_NAME)
    with portalocker.Lock(pool_lock_path, timeout=1) as _:
      write_gene(gene_data, gene_name, RUN_NAME)

    count = int(all_args['count'])
    if count >= 5:
      sys.exit()

    p = subprocess.Popen(["python3", "popen_test.py", "--call_type=server_callback", f"--count={count}"])

  elif call_type == "server_callback":
    count = int(all_args['count'])
    count += 1

    # Lock pool during gene creation
    pool_lock_path = file_path(RUN_NAME, POOL_LOCK_NAME)
    with portalocker.Lock(pool_lock_path, timeout=1) as _:

      # Init alg (loads gene pool)
      alg = Algorithm(RUN_NAME, GENE_SHAPE, MUTATION_RATE, NUM_GENES)

      # Fetch next gene for testing
      gene_name, _ = alg.fetch_gene()

    p = subprocess.Popen(["python3", "popen_test.py", "--call_type=run_client", f"--gene_name={gene_name}",
                          f"--count={count}"])

  else:
    print(f"error, improper call_type: {call_type}")
