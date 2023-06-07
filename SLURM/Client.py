import time
import ast
from os import mkdir, rmdir
from os.path import join as file_path, isdir
from filelock import FileLock
import os
import argparse
import pickle
import numpy as np
import sys
import subprocess
import hashlib

POOL_DIR = "pool"
LOCK_DIR = "locks"

# TODO: Temporary import, move these to a different independent file
from Algorithm import write_gene, load_gene   # Shouldn't need deletes

class Client():
  def __init__(self, run_name: str, gene_name: str):
    self.run_name = run_name
    self.gene_name = gene_name
    self.gene_data = load_gene(gene_name, run_name)

  # Run model
  def run(self):
    gene = self.gene_data['gene']
    fitness = sum([-(i**2) for i in gene])
    return fitness


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:  # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())
  # call_type = all_args.pop('call_type')

  RUN_NAME = "test_dir"
  GENE_SHAPE = 10
  MUTATION_RATE = 0.2
  NUM_GENES = 10

  # Run gene
  gene_name = '3b119ef98ba88716540f3e053b57afeaee0cc5f4671681d9af6464017929b405'
  client = Client(RUN_NAME, gene_name)
  fitness = client.run()

  # Return fitness (by writing to files)
  gene_data = client.gene_data
  gene_data['fitness'] = fitness
  gene_data['status'] = 'tested'
  write_gene(gene_data, gene_name, RUN_NAME)
  count = int(all_args['count'])
  p = subprocess.Popen(["python3", "popen_test.py", "--call_type=server_callback", f"--count={count}"])

  # gene_name = '3b119ef98ba88716540f3e053b57afeaee0cc5f4671681d9af6464017929b405'

