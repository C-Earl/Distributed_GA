import subprocess
import pickle
from os.path import join as file_path
from filelock import FileLock
import sys
import argparse
import time
import numpy as np
import os
from Algorithm import Algorithm, write_gene

POOL_DIR = "pool"
LOCK_DIR = "locks"
TEST_DIR = "test_dir"
TEST_GENE_NAME = "test_gene"
CLIENT_RUNNER = "run_client.sh"
SERVER_CALLBACK = "run_server.sh"

def test_marker(id):
  with open(f"{id}.txt", 'w') as file:
    file.write("hi")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())
  call_type = all_args.pop('call_type')

  if call_type == "init":
    for i in range(10):
      # Initialize directories
      # Initialize starter genes
      p = subprocess.Popen(["python3", "popen_test.py", "--call_type=run_client", "--count=0"])

  elif call_type == "run_client":
    time.sleep(3)

    # Run gene

    # Return fitness
    count = int(all_args['count'])
    # write_gene({"count":count, "gene_data": np.random.rand(100)}, name=TEST_GENE_NAME, run_name=TEST_DIR)
    p = subprocess.Popen(["python3", "popen_test.py", "--call_type=server_callback", f"--count={count}"])

  elif call_type == "server_callback":
    count = int(all_args['count'])
    count += 1
    if count >= 5:
      sys.exit()

    # Init alg (loads gene pool)
    # run_name = all_args['run_name']
    # gene_shape = all_args['gene_shape']
    # mutation_rate = all_args['mutation_rate']
    # alg = Algorithm(run_name, gene_shape, mutation_rate)
    #
    # # Generate gene with alg
    # new_gene = alg.create_gene()
    # gene_name = new_gene['name']

    # Send gene to client
    # p = subprocess.Popen(["python3", "popen_test.py", "--call_type=run_client", f"--count={count}",
                          # f"--gene_name={gene_name}"])
    p = subprocess.Popen(["python3", "popen_test.py", "--call_type=run_client", f"--count={count}"])

  else:
    print(f"error, improper call_type: {call_type}")
