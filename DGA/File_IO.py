import os
from os.path import join as file_path
from DGA.Gene import Gene, Genome, Parameters
import pickle
import json
import jsbeautifier
import numpy as np
import copy
import ast

# Constants for filesystem
POOL_DIR = "pool"
LOG_DIR = "logs"
ARGS_FOLDER = "run_args"
POOL_LOG_NAME = "POOL_LOG.log"
POOL_LOCK_NAME = "POOL_LOCK.lock"
DATA_LOCK_NAME = "DATA_LOCK.lock"
RUN_STATUS_NAME_JSON = "RUN_STATUS.json"
RUN_STATUS_NAME_PKL = "RUN_STATUS.pkl"
ERROR_LOG_NAME = "ERROR_LOG.log"

# Transform any non-json compatible types
def jsonify(d: dict):
  for k, v in d.items():
    if isinstance(v, dict):   # Recursively jsonify
      d[k] = jsonify(v)
    if isinstance(v, np.ndarray):
      d[k] = v.tolist()
    if isinstance(v, type):
      d[k] = str(v)
    if isinstance(v, Genome):
      d[k] = v.to_json()
    if isinstance(v, Gene):
      d[k] = v.to_json()
  return d


# Arguments passed to model process are first written to file. This function writes them.
def write_model_args_to_file(model_id: int, **kwargs):
  kwargs['model_id'] = model_id

  # Write to pkl
  args_path = file_path(kwargs['run_name'], ARGS_FOLDER, f"model{model_id}_args.pkl")
  with open(args_path, 'wb') as args_file:
    pickle.dump(kwargs, args_file)

  # Write to json
  args_path = file_path(kwargs['run_name'], ARGS_FOLDER, f"model{model_id}_args.json")
  kwargs_json = jsonify(copy.deepcopy(kwargs))
  with open(args_path, 'w') as args_file:
    json.dump(kwargs_json, args_file, indent=2)


# Server writes args to file, model process the loads with this function
def load_model_args_from_file(model_id: int, run_name: str):
  args_path = file_path(run_name, ARGS_FOLDER, f"model{model_id}_args.pkl")   # Only load from pickle file
  with open(args_path, 'rb') as args_file:
    return pickle.load(args_file)


# Take gene and write it to a file. Returns file name and written data
# def create_gene_file(run_name: str, gene_name: str, gene: np.ndarray | dict):
#   gene_data = {'gene': gene, 'fitness': None, 'test_state': 'being tested'}
#   write_gene_file(gene_data, gene_name, run_name)


# Write gene to file
def write_gene_file(run_name: str, gene_name: str, gene: dict):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, gene_name) + ".pkl"
  with open(gene_path, 'wb') as gene_file:
    pickle.dump(gene, gene_file)


# Load gene from file
def load_gene_file(run_name: str, gene_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, gene_name) + ".pkl"
  with open(gene_path, 'rb') as gene_file:
    gene = pickle.load(gene_file)
  return gene


# Delete gene file
def delete_gene_file(run_name: str, gene_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  gene_path = file_path(pool_path, gene_name) + ".pkl"
  os.remove(gene_path)
  return True


# Read status of run to file (only read pickle file)
def read_run_status(run_name: str):
  status_path = file_path(run_name, RUN_STATUS_NAME_PKL)
  with open(status_path, 'rb') as status_file:
    run_status = pickle.load(status_file)
  return run_status


# Write status of run to pickle file for loading, and json file for human readability
def write_run_status(run_name: str, status: dict):
  # Write to pkl
  status_path = file_path(run_name, RUN_STATUS_NAME_PKL)
  with open(status_path, 'wb') as status_file:
    pickle.dump(status, status_file)

  # Write to json
  status_copy = jsonify(copy.deepcopy(status))
  status_path = file_path(run_name, RUN_STATUS_NAME_JSON)
  options = jsbeautifier.default_options()
  options.indent_size = 2
  with open(status_path, 'w') as status_file:
    status_file.write(jsbeautifier.beautify(json.dumps(status_copy), options))


# Write to model log file
def write_log(run_name: str, model_id: int, log: dict | np.ndarray):
  log_path = file_path(run_name, LOG_DIR, f'model_{str(model_id)}' + ".log")
  with open(log_path, 'a+') as log_file:
    log_file.write(json.dumps(log) + "\n")    # Not json.dump because want each log on a new line


def read_log(run_name: str, model_id: int):
  log_path = file_path(run_name, LOG_DIR, f'model_{str(model_id)}' + ".log")
  with open(log_path, 'r') as log_file:
    logs = log_file.readlines()
  for i in range(len(logs)):
    logs[i] = ast.literal_eval(logs[i])
  return logs


def write_pool_log(run_name: str, pool_log: dict):
  pool_log_copy = jsonify(copy.deepcopy(pool_log))
  log_path = file_path(run_name, LOG_DIR, POOL_LOG_NAME)
  with open(log_path, 'a') as log_file:
    log_file.write(json.dumps(pool_log_copy) + "\n")    # Not json.dump because want each log on a new line

def write_error_log(run_name: str, error_log: dict):
  error_log_copy = jsonify(error_log)
  log_path = file_path(run_name, LOG_DIR, ERROR_LOG_NAME)
  with open(log_path, 'a') as log_file:
    log_file.write(json.dumps(error_log_copy) + "\n")