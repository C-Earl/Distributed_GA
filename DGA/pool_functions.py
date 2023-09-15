import os
from os.path import join as file_path
import pickle
import json
import numpy as np

# Constants for filesystem
POOL_DIR = "pool"
LOG_DIR = "logs"
ARGS_FOLDER = "run_args"
POOL_LOCK_NAME = "POOL_LOCK.lock"
DATA_LOCK_NAME = "DATA_LOCK.lock"
RUN_STATUS_NAME = "RUN_STATUS.json"


# Arguments passed to client process are first written to file. This function writes them.
def write_args_to_file(client_id: int, **kwargs):
  args_path = file_path(kwargs['run_name'], ARGS_FOLDER, f"client{client_id}_args.json")
  kwargs['client_id'] = client_id
  with open(args_path, 'w') as args_file:
    json.dump(kwargs, args_file)


# Server writes args to file, client process the loads with this function
def load_args_from_file(client_id: int, run_name: str):
  args_path = file_path(run_name, ARGS_FOLDER, f"client{client_id}_args.json")
  with open(args_path, 'r') as args_file:
    return json.load(args_file)


# Write gene to file
def write_gene(gene: dict, name: str, run_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  # try:
  gene_path = file_path(pool_path, name) + ".pkl"
  # except TypeError:
  #   print(pool_path, name, run_name, gene)
  with open(gene_path, 'wb') as gene_file:
    pickle.dump(gene, gene_file)


# Load gene from file
def load_gene(name: str, run_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, name) + ".pkl"
  with open(gene_path, 'rb') as gene_file:
    gene = pickle.load(gene_file)
  return gene


# Delete gene file
def delete_gene(name: str, run_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, name) + ".pkl"
  os.remove(gene_path)
  return True


# Create new run status
def create_run_status(run_name: str, init_status: dict):
  status_path = file_path(run_name, RUN_STATUS_NAME)
  with open(status_path, 'w') as status_file:
    json.dump(init_status, status_file)
    # pickle.dump(init_status, status_file)


# Read status of run to file
def read_run_status(run_name: str):
  status_path = file_path(run_name, RUN_STATUS_NAME)
  with open(status_path, 'r') as status_file:
    run_status = json.load(status_file)
  return run_status


# Write status of run to file
def write_run_status(run_name: str, status: dict):
  status_path = file_path(run_name, RUN_STATUS_NAME)
  with open(status_path, 'w') as status_file:
    json.dump(status, status_file)
    # pickle.dump(status, status_file)


# Write to client log file
def write_log(run_name: str, client_id: int, log: dict | type(np.array)):
  log_path = file_path(run_name, LOG_DIR, f'client_{str(client_id)}' + ".log")
  with open(log_path, 'a') as log_file:
    log_file.write(json.dumps(log) + "\n")
