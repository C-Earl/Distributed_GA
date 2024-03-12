import logging
import copy
import os.path
import subprocess
from os.path import join as file_path
import numpy as np
import sys
import argparse
import time
from DGA.File_IO import write_params_file, load_params_file, POOL_DIR, LOG_DIR, RUN_INFO_DIR, POOL_LOCK_NAME, write_log, \
  write_model_args_to_file, load_model_args_from_file, \
  delete_params_file, write_error_log, save_model, load_model, save_algorithm, load_algorithm, load_params_file_async, \
  load_algorithm_async, delete_params_file_async, save_algorithm_async, load_agent_job_ID, SERVER_LOG_DIR, ALG_DIR
from DGA.Algorithm import Genetic_Algorithm_Base as Algorithm
from DGA.Model import Model


# https://stackoverflow.com/questions/17075071/is-there-a-python-method-to-access-all-non-private-and-non-builtin-attributes-of
# Helper function to get all public attributes of an object
def list_public_attributes(input_var):
  return [k for k, v in vars(input_var).items() if
          not (k.startswith('_') or callable(v))]


class Server:
  def __init__(self, run_name: str, algorithm: Algorithm, model: Model,
               num_parallel_processes: int, call_type: str = 'init',
               data_path: str = None, log_pool: int = -1, **kwargs):

    self.run_name = run_name  # Name of run (used for folder name)
    self.num_parallel_processes = num_parallel_processes
    self.data_path = data_path  # Location of data folder (if needed, for async loading)
    self.server_file_path = os.path.abspath(__file__)  # Note: CWD not the same as DGA folder
    self.log_pool = log_pool  # Log pool-state every n params (-1 for no logging)

    # Switch for handling model, server, or run initialization
    if call_type == "init":

      algorithm_type = type(algorithm)
      model_type = type(model)
      genome_type = type(algorithm.genome)
      gene_types = set([type(gene) for gene in algorithm.genome.values()])

      # Retrieve args (set by the user) passed to Model and Algorithm objects
      algorithm_args = list_public_attributes(algorithm)
      algorithm_args = {key: algorithm.__dict__[key] for key in algorithm_args}

      # Define paths to model and algorithm files (used for loading in subprocess)
      self.algorithm_path = os.path.abspath(sys.modules[algorithm_type.__module__].__file__)
      self.model_path = os.path.abspath(sys.modules[model_type.__module__].__file__)
      self.genome_path = os.path.abspath(sys.modules[genome_type.__module__].__file__)
      self.gene_paths = [os.path.abspath(sys.modules[gene_type.__module__].__file__) for gene_type in gene_types]
      self.algorithm_name = algorithm_type.__name__
      self.model_name = model_type.__name__
      self.genome_name = genome_type.__name__
      self.gene_names = [gene_type.__name__ for gene_type in gene_types]

      self.init(algorithm_type, algorithm_args, model, **kwargs)

    elif call_type == "run_model":
      # Reload paths to model and algorithm files (used for loading in subprocess)
      self.algorithm_path = kwargs.pop('algorithm_path')
      self.model_path = kwargs.pop('model_path')
      self.genome_path = kwargs.pop('genome_path')
      self.gene_paths = kwargs.pop('gene_paths')
      self.algorithm_name = kwargs.pop('algorithm_name')
      self.model_name = kwargs.pop('model_name')
      self.genome_name = kwargs.pop('genome_name')
      self.gene_names = kwargs.pop('gene_names')
      self.run_model(**kwargs)

    elif call_type == "server_callback":
      # Reload paths to model and algorithm files (used for loading in subprocess)
      self.algorithm_path = kwargs.pop('algorithm_path')
      self.model_path = kwargs.pop('model_path')
      self.genome_path = kwargs.pop('genome_path')
      self.gene_paths = kwargs.pop('gene_paths')
      self.algorithm_name = kwargs.pop('algorithm_name')
      self.model_name = kwargs.pop('model_name')
      self.genome_name = kwargs.pop('genome_name')
      self.gene_names = kwargs.pop('gene_names')
      self.server_callback(**kwargs)

    else:
      raise Exception(f"error, improper call_type: {call_type}")

  # Initialize run
  def init(self, algorithm_type: type, algorithm_args: dict, model: Model, **kwargs):

    # Make run directories
    # TODO: This prevents continuing runs.
    os.makedirs(file_path(self.run_name, POOL_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, LOG_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, LOG_DIR, SERVER_LOG_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, RUN_INFO_DIR), exist_ok=True)
    os.makedirs(file_path(self.run_name, RUN_INFO_DIR, ALG_DIR), exist_ok=True)

    # Generate initial params
    alg = algorithm_type(run_name=self.run_name, **algorithm_args)
    original_pool = copy.deepcopy(alg.pool)
    init_params = alg.initialize_pool(self.num_parallel_processes)
    # for i in range(self.num_parallel_processes):
      # init_params.append(alg.fetch_params(agent_id=i)[0])  # Don't need status on init, just params
      # init_param = alg.spawn(iteration=0)
      # init_param_name =
      # init_params.append()

    # Update pool files (Parameter files)
    final_pool = alg.pool
    self.update_pool(original_pool, final_pool)

    # Save algorithm & model to files
    save_algorithm_async(self.run_name, alg)
    save_model(self.run_name, model)

    # Call models to run initial params
    for i, p_name in enumerate(init_params.keys()):
      self.make_call(i, p_name, "run_model", **kwargs)

  # Run model with given params
  def run_model(self, **kwargs):
    # Setup model
    params_name = kwargs['params_name']
    params = load_params_file_async(self.run_name, params_name)  # All params info (inc. fitness, etc.)
    model = load_model(self.run_name)

    # Load data
    model.load_data()

    # Test params
    runtime_args = params['runtime_args'] if 'runtime_args' in params else {}
    fitness = model.run(params, **runtime_args, **kwargs)

    # Return fitness (by writing to files)
    params.set_fitness(fitness)
    params.set_tested(True)
    pool_lock_path = file_path(self.run_name, POOL_LOCK_NAME)
    # with portalocker.Lock(pool_lock_path, timeout=10) as _:
    write_params_file(self.run_name, params_name, params)
    write_log(self.run_name, kwargs['agent_id'], model.logger(params))

    # Callback server
    self.make_call(call_type="server_callback", **kwargs)  # Other args contained in kwargs

  # Callback server to get next params
  def server_callback(self, **kwargs):

    # Setup algorithm by loading from file
    # - updates pool and history based on files which other agents may have changed
    while True:
      alg = load_algorithm_async(self.run_name, buffer_length=self.num_parallel_processes)
      if alg.history is not None:
        alg.history.update(alg.load_history(async_=True))
      alg.pool.update(alg.load_pool(async_=True))
      alg.pool.update_subpools()

      # Copy pool for later comparison when updating files
      orginal_pool = copy.deepcopy(alg.pool)

      # Check if run is complete
      if alg.end_condition():
        sys.exit()

      # Generate next params for testing
      params_name, success = alg.fetch_params(agent_id=kwargs['agent_id'])

      # Update status & break if fetch was success, otherwise try again
      if success:

        # Update pool files (pool modified by alg)
        final_pool = alg.pool
        self.update_pool(orginal_pool, final_pool)

        # Update pool log & save alg to file
        save_algorithm_async(self.run_name, alg)

        break
      else:
        time.sleep(np.random.rand())

    # Remove old params_name from args, and send new params to model
    kwargs.pop('params_name')
    self.make_call(call_type="run_model", params_name=params_name, **kwargs)

  # Update pool files (files with Parameter data)
  def update_pool(self, original_pool: dict, new_pool: dict):
    # Injective check from new_pool to original_pool
    for params_name in new_pool.keys():
      if params_name not in original_pool.keys():  # If it's a newly added params
        new_params_data = new_pool[params_name]
        write_params_file(self.run_name, params_name, new_params_data)

    # Injective check from original_pool to new_pool
    for params_name in original_pool.keys():
      if params_name not in new_pool.keys():  # If it's a removed params
        delete_params_file_async(self.run_name, params_name)

  # Save important args to file and run next phase (callback or run_model)
  # Saves here are per-model, so that each model can run independently
  def make_call(self,
                agent_id: int,
                params_name: str,
                call_type: str,
                **kwargs):
    write_model_args_to_file(agent_id=agent_id,
                             params_name=params_name,
                             call_type=call_type,  # callback or run_model
                             run_name=self.run_name,
                             algorithm_path=self.algorithm_path,
                             algorithm_name=self.algorithm_name,
                             model_path=self.model_path,
                             model_name=self.model_name,
                             genome_path=self.genome_path,
                             genome_name=self.genome_name,
                             gene_paths=self.gene_paths,
                             gene_names=self.gene_names,
                             num_parallel_processes=self.num_parallel_processes,
                             data_path=self.data_path,
                             log_pool=self.log_pool,
                             **kwargs)

    # Save

    # Run command according to OS
    # TODO: SOMEONE DO THIS FOR MAC PLEASE
    if sys.platform == "linux":
      p = subprocess.Popen(["python3", self.server_file_path, f"--run_name={self.run_name}", f"--agent_id={agent_id}"])
    elif sys.platform == "win32":
      p = subprocess.Popen(["python", self.server_file_path, f"--run_name={self.run_name}", f"--agent_id={agent_id}"],
                           shell=True)
    elif sys.platform == "darwin":
      pass  # MAC HANDLING


# Main function catches server-callbacks & runs models
# NOTE:
# When users define their own classes, they *must not* run the Server from the same file. A separate main.py file
# should be made for this. The reason is a technical issue with Pickle and Python imports.
# Source for explanation: https://stackoverflow.com/questions/50465106/attributeerror-when-reading-a-pickle-file
#     - Briefly: When starting a run, if the custom objects (like Algorithms, Genomes, etc.) are loaded from a
#       if __name__ == '__main__': block, then the pickle file will not be able to load the custom objects after saving(see link
#       for why). If the run code is not contained in __main__, then everytime we re-import the custom objects to reload
#       them from the disk, the run code will be called recursively and probably crash your computer!
# TODO: Make this safer ^
if __name__ == '__main__':
  parser_ = argparse.ArgumentParser()
  parser_.add_argument('--agent_id', type=int)
  parser_.add_argument('--run_name', type=str)
  args_ = parser_.parse_args()

  # Load args from file
  all_args = load_model_args_from_file(args_.agent_id, args_.run_name)

  # Establish location of Algorithm and Model classes & add them to python path
  algorithm_path_ = all_args['algorithm_path']
  algorithm_name_ = all_args['algorithm_name']
  model_path_ = all_args['model_path']
  model_name_ = all_args['model_name']
  genome_path_ = all_args['genome_path']
  genome_name_ = all_args['genome_name']
  gene_paths_ = all_args['gene_paths']
  gene_names_ = all_args['gene_names']
  server_path_ = os.path.abspath(__file__)  # Get absolute path to current location on machine
  base_path_ = '/'.join(server_path_.split('/')[0:-2])  # Get path to "./Distributed_GA" ie. base folder
  alg_module_path_ = file_path(base_path_, '/'.join(algorithm_path_.split('/')[0:-1]))
  model_module_path_ = file_path(base_path_, '/'.join(model_path_.split('/')[0:-1]))
  genome_module_path_ = file_path(base_path_, '/'.join(genome_path_.split('/')[0:-1]))
  # model_module_name_ = model_module_path_.split('/')[-1][:-3]
  sys.path.append(alg_module_path_)
  sys.path.append(model_module_path_)
  sys.path.append(genome_module_path_)
  for gene_path in gene_paths_:
    sys.path.append(gene_path)

  # Create Algorithm and Model objects
  alg_module_name_ = algorithm_path_.split('/')[-1][:-3]
  model_module_name_ = model_path_.split('/')[-1][:-3]
  genome_module_name_ = genome_path_.split('/')[-1][:-3]
  alg_module_ = __import__(alg_module_name_)
  algorithm_ = getattr(alg_module_, algorithm_name_)
  model_module_ = __import__(model_module_name_)
  model_ = getattr(model_module_, model_name_)
  genome_module_ = __import__(genome_module_name_)
  genome_ = getattr(genome_module_, genome_name_)
  genes_ = [getattr(__import__(gene_name_), gene_name_) for gene_name_ in gene_names_]
  all_args['algorithm'] = None  # Can't load until obtained file lock
  all_args['model'] = None

  # Create logger for server
  logging.basicConfig(filename=str(file_path(all_args['run_name'], LOG_DIR, SERVER_LOG_DIR, f"AGENT_{args_.agent_id}.log")),
                      encoding='utf-8',
                      level=logging.DEBUG)

  # Run server protocol with bash kwargs
  try:
    Server(**all_args)
  except:
    logging.exception(f" Error on Agent {args_.agent_id}:")
    raise
