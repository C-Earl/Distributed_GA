import copy
import os.path
from os.path import join as file_path
from os import system as cmd
import sys
import argparse
from portalocker.exceptions import LockException
from DGA.File_IO import write_model_args_to_file, load_model_args_from_file, write_error_log, load_algorithm
from DGA.Algorithm import Genetic_Algorithm_Base as Algorithm
from DGA.Model import Model
from DGA.Server import Server


class Server_SLURM(Server):
  def __init__(self, run_name: str, algorithm: Algorithm | type, model: Model | type,
               num_parallel_processes: int, sbatch_script: str, callback_sbatch_script: str = None, call_type: str = 'init',
               data_path: str = None, log_pool: int = -1, **kwargs):
    self.sbatch_script = sbatch_script
    self.callback_sbatch_script = callback_sbatch_script
    super().__init__(run_name, algorithm, model, num_parallel_processes, call_type, data_path, log_pool, **kwargs)

  def server_callback(self, **kwargs):
    try:
      super().server_callback(**kwargs)
    except LockException:
      # Ignore lock, force param creation
      alg = load_algorithm(self.run_name)
      if alg.history is not None:
        alg.history.update(alg.load_history())
      alg.pool.update(alg.load_pool())
      alg.pool.update_subpools()

      orginal_pool = copy.deepcopy(alg.pool)

      if alg.end_condition():
        sys.exit()

      # Generate next params for testing
      if len(alg.valid_params) > alg.num_parents:
        new_params = alg.breed()
      else:
        new_params = alg.spawn()

      # Save new params to file
      final_pool = alg.pool
      self.update_pool(orginal_pool, final_pool)

      kwargs.pop('params_name')
      params_name = str(hash(new_params))
      self.make_call(call_type='run_model', params_name=params_name, **kwargs)


  # Save args for model, and make sbatch calls
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
                              sbatch_script=self.sbatch_script,   # Needed for SLURM
                              log_pool=self.log_pool,
                              **kwargs)

    # Call sbatch script
    if call_type == 'run_model':
      server_path_ = os.path.abspath(__file__)  # Get absolute path to current location on machine
      cmd(f"sbatch {self.sbatch_script} {agent_id} {self.run_name} {server_path_}")
    elif call_type == 'server_callback':     # If true, means already on node, no need to make new node
      try:
        self.server_callback(**kwargs, agent_id=agent_id, params_name=params_name)
      except LockException:
        server_path_ = os.path.abspath(__file__)
        cmd(f"sbatch {self.sbatch_script} {agent_id} {self.run_name} {server_path_}")

    # elif call_type == 'server_callback':
    #   # Check if need to call new node for callback. If not, run callback on current node
    #   if self.callback_sbatch_script is not None:
    #     cmd(f"sbatch {self.callback_sbatch_script} {agent_id} {self.run_name} {server_path_}")
    #   else:
    #     self.server_callback(**kwargs, agent_id=agent_id)

# Main function catches server-callbacks & runs models
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
  base_path_ = '/'.join(server_path_.split('/')[0:-2])    # Get path to "./Distributed_GA" ie. base folder
  alg_module_path_ = file_path(base_path_, '/'.join(algorithm_path_.split('/')[0:-1]))
  model_module_path_ = file_path(base_path_, '/'.join(model_path_.split('/')[0:-1]))
  genome_module_path_ = file_path(base_path_, '/'.join(genome_path_.split('/')[0:-1]))
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
  all_args['algorithm'] = None      # Can't load until obtained file lock
  all_args['model'] = None

  # Run server protocol with bash kwargs
  try:
    Server_SLURM(**all_args)
  except Exception as e:
    write_error_log(all_args['run_name'], {'error': str(e)})
    raise e
