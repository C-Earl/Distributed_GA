import datetime
import os
import time
from os.path import join as file_path
from DGA.Gene import Gene, Genome, Parameters
from DGA.Model import Model
from DGA.Pool import Pool
import pickle
import json
import jsbeautifier
import numpy as np
import copy
import ast

# Constants for filesystem
POOL_DIR = "pool"
LOG_DIR = "logs"
ALG_DIR = "algorithm_buffer"
RUN_INFO = "run_info"
POOL_LOG_NAME = "POOL_LOG.log"
POOL_LOCK_NAME = "POOL_LOCK.lock"
DATA_LOCK_NAME = "DATA_LOCK.lock"
RUN_STATUS_NAME_JSON = "RUN_STATUS.json"
RUN_STATUS_NAME_PKL = "RUN_STATUS.pkl"
ERROR_LOG_NAME = "ERROR_LOG.log"
AGENT_NAME = "AGENT"

# Transform any non-json compatible types
def jsonify(d: dict):
  if isinstance(d, dict):
    for k, v in d.items():
      d[k] = jsonify(v)
  elif isinstance(d, list):
    return [jsonify(i) for i in d]
  elif isinstance(d, np.ndarray):
    return d.tolist()
  elif isinstance(d, type):
    return str(d)
  elif isinstance(d, Genome):
    return d.to_json()
  elif isinstance(d, Gene):
    return d.to_json()
  elif isinstance(d, Model):
    return d.args_to_json()
  return d


# Arguments passed to model process are first written to file. This function writes them.
def write_model_args_to_file(agent_id: int, **kwargs):
  kwargs['agent_id'] = agent_id

  # Write to pkl
  args_path = file_path(kwargs['run_name'], RUN_INFO, f"{AGENT_NAME}_{agent_id}_args.pkl")
  with open(args_path, 'wb') as args_file:
    pickle.dump(kwargs, args_file)

  # Write to json
  args_path = file_path(kwargs['run_name'], RUN_INFO, f"{AGENT_NAME}_{agent_id}_args.json")
  kwargs_json = jsonify(copy.deepcopy(kwargs))
  with open(args_path, 'w') as args_file:
    json.dump(kwargs_json, args_file, indent=2)


# Server writes args to file, model process the loads with this function
def load_model_args_from_file(agent_id: int, run_name: str):
  args_path = file_path(run_name, RUN_INFO, f"{AGENT_NAME}_{agent_id}_args.pkl")   # Only load from pickle file
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


def load_params_file_async(run_name: str, params_name: str):
  while True:
    try:
      return load_params_file(run_name, params_name)
    except EOFError:
      time.sleep(np.random.rand() * 0.1)
      with open(file_path(run_name, LOG_DIR, ERROR_LOG_NAME), 'a') as log_file:
        log_file.write(f"EOFError loading params {params_name}\n")
    except pickle.UnpicklingError:
      time.sleep(np.random.rand() * 0.1)
      with open(file_path(run_name, LOG_DIR, ERROR_LOG_NAME), 'a') as log_file:
        log_file.write(f"Error unpickling params file {params_name}\n")
    except FileNotFoundError:
      return None


# Delete params file
def delete_params_file(run_name: str, params_name: str):
  pool_path = file_path(run_name, POOL_DIR)
  params_path = file_path(pool_path, params_name) + ".pkl"
  os.remove(params_path)
  return True


def delete_params_file_async(run_name: str, params_name: str):
  try:
    delete_params_file(run_name, params_name)
  except FileNotFoundError:
    return False


# # Read status of run to file (only read pickle file)
# def read_run_status(run_name: str):
#   status_path = file_path(run_name, RUN_STATUS_NAME_PKL)
#   with open(status_path, 'rb') as status_file:
#     run_status = pickle.load(status_file)
#   return run_status
#
#
# # Write status of run to pickle file for loading, and json file for human readability
# def write_run_status(run_name: str, status: dict):
#   # Write to pkl
#   status_path = file_path(run_name, RUN_STATUS_NAME_PKL)
#   with open(status_path, 'wb') as status_file:
#     pickle.dump(status, status_file)
#
#   # Write to json
#   status_copy = jsonify(copy.deepcopy(status))
#   status_path = file_path(run_name, RUN_STATUS_NAME_JSON)
#   options = jsbeautifier.default_options()
#   options.indent_size = 2
#   with open(status_path, 'w') as status_file:
#     status_file.write(jsbeautifier.beautify(json.dumps(status_copy), options))


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


def load_history_async(run_name: str):
  while True:
    try:
      return load_history(run_name)
    except EOFError:
      time.sleep(np.random.rand() * 0.1)
      with open(file_path(run_name, LOG_DIR, ERROR_LOG_NAME), 'a') as log_file:
        log_file.write(f"EOFError loading history\n")


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


def save_model(run_name: str, model: Model):
  model_path = file_path(run_name, RUN_INFO, f"model.pkl")
  with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)


def load_model(run_name: str):
  model_path = file_path(run_name, RUN_INFO, f"model.pkl")
  with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
  return model


def save_algorithm(run_name: str, algorithm):
  save_name = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
  alg_path = file_path(run_name, RUN_INFO, ALG_DIR, save_name)
  with open(alg_path, 'wb') as alg_file:
    pickle.dump(algorithm, alg_file)


def load_algorithm(run_name: str):
  while True:
    alg_saves = os.listdir(file_path(run_name, RUN_INFO, ALG_DIR))
    alg_saves.sort()
    alg_path = file_path(run_name, RUN_INFO, ALG_DIR, alg_saves[-1])
    try:
      with open(alg_path, 'rb') as alg_file:
        alg = pickle.load(alg_file)
        break

    # File is being written to, wait a bit and try again
    except EOFError as e:
      with open(file_path(run_name, LOG_DIR, ERROR_LOG_NAME), 'a') as log_file:
        log_file.write(f"EOF error with algorithm: {e}\n")
      time.sleep(np.random.rand() * 0.1)

    # File not found, wait a bit and try again
    except FileNotFoundError as e:
      with open(file_path(run_name, LOG_DIR, ERROR_LOG_NAME), 'a') as log_file:
        log_file.write(f"File Not Found error with algorithm: {e}\n")
      time.sleep(np.random.rand() * 0.1)

    # Corrupt file detected, remove it & try to load again
    except Exception as e:
      with open(file_path(run_name, LOG_DIR, ERROR_LOG_NAME), 'a') as log_file:
        log_file.write(f"Error loading algorithm: {e}\n")
      try:
        os.remove(alg_path)
      except FileNotFoundError:
        pass

  if len(alg_saves) > 10:
    num_remove = len(alg_saves) - 10
    for i in range(num_remove):
      try:
        os.remove(file_path(run_name, RUN_INFO, ALG_DIR, alg_saves[i]))
      except FileNotFoundError:
        pass
  return alg


def load_algorithm_async(run_name: str):
  # while True:
  #   try:
  return load_algorithm(run_name)
    # except EOFError:
    #   time.sleep(np.random.rand() * 0.1)
    # except pickle.UnpicklingError as e:
    #   time.sleep(np.random.rand() * 0.1)
    #   with open(file_path(run_name, LOG_DIR, ERROR_LOG_NAME), 'a') as log_file:
    #     log_file.write(f"Unpickling error with algorithm: {e}\n")
