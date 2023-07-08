import argparse
import os
import sys
from DGA.pool_functions import load_gene
from os.path import join as file_path

POOL_DIR = "pool"

def main(run_name: str):
  # Load gene pool
  pool = {}
  pool_path = file_path(run_name, "pool")
  for root, dirs, files in os.walk(pool_path):
    for file in files:
      file_name = file.split('.')[0]  # This will be unique hash of the gene
      gene = load_gene(file_name, run_name)
      pool[file_name] = gene

  for gene_name, gene in pool.items():
    print(f"Gene: {gene_name[0:10]}...\t\t" + f"Fitness: {gene['fitness']}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())

  # If no arguments are passed, use manual inputs
  if len(all_args.items()) == 0:

    ### Manual Inputs ###
    all_args['run_name'] = "example_run_name"

  main(**all_args)
