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
AGENT_NAME = "AGENT"

# Transform any non-json compatible types
def jsonify(d: dict):
  for k, v in d.items():
    if isinstance(v, dict):   # Recursively jsonify
      d[k] = jsonify(v)
    elif isinstance(v, np.ndarray):
      d[k] = v.tolist()
    elif isinstance(v, type):
      d[k] = str(v)
    elif isinstance(v, Genome):
      d[k] = v.to_json()
    elif isinstance(v, Gene):
      d[k] = v.to_json()
    if (isinstance(v, Parameters)):
      print("hi")
  # print(d)
  return d


# Arguments passed to model process are first written to file. This function writes them.
def write_model_args_to_file(agent_id: int, **kwargs):
  kwargs['agent_id'] = agent_id

  # Write to pkl
  args_path = file_path(kwargs['run_name'], ARGS_FOLDER, f"{AGENT_NAME}_{agent_id}_args.pkl")
  with open(args_path, 'wb') as args_file:
    pickle.dump(kwargs, args_file)

  # Write to json
  args_path = file_path(kwargs['run_name'], ARGS_FOLDER, f"{AGENT_NAME}_{agent_id}_args.json")
  kwargs_json = jsonify(copy.deepcopy(kwargs))
  with open(args_path, 'w') as args_file:
    json.dump(kwargs_json, args_file, indent=2)


# Server writes args to file, model process the loads with this function
def load_model_args_from_file(agent_id: int, run_name: str):
  args_path = file_path(run_name, ARGS_FOLDER, f"{AGENT_NAME}_{agent_id}_args.pkl")   # Only load from pickle file
  with open(args_path, 'rb') as args_file:
    return pickle.load(args_file)


# Write params to file
def write_params_file(run_name: str, params_name: str, params: Parameters):
  pool_path = file_path(run_name, POOL_DIR)
  params_path = file_path(pool_path, params_name) + ".pkl"
  with open(params_path, 'wb') as params_file:
    pickle.dump(params, params_file)


# Load params from file
def load_params_file(run_name: str, params_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  params_path = file_path(pool_path, params_name) + ".pkl"
  with open(params_path, 'rb') as params_file:
    params = pickle.load(params_file)
  return params


# Delete params file
def delete_params_file(run_name: str, params_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  params_path = file_path(pool_path, params_name) + ".pkl"
  os.remove(params_path)
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
def write_log(run_name: str, agent_id: int, log: dict | np.ndarray):
  log_path = file_path(run_name, LOG_DIR, f'{AGENT_NAME}_{str(agent_id)}' + ".log")
  with open(log_path, 'a+') as log_file:
    log_file.write(json.dumps(log) + "\n")    # Not json.dump because want each log on a new line


def read_log(run_name: str, agent_id: int):
  log_path = file_path(run_name, LOG_DIR, f'{AGENT_NAME}_{str(agent_id)}' + ".log")
  with open(log_path, 'r') as log_file:
    logs = log_file.readlines()
  for i in range(len(logs)):
    logs[i] = ast.literal_eval(logs[i])
  return logs


def load_history(run_name: str):
  history = {}
  log_folder = file_path(run_name, LOG_DIR)
  for root, dirs, files in os.walk(log_folder):
    for file in files:
      if file.endswith(".log") and not file == 'ERROR_LOG.log':
        agent_id = int(file.split('_')[1].split('.')[0])
        history[agent_id] = read_log(run_name, agent_id)
  return history


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
