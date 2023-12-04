import os.path
from os.path import join as file_path
from os import system as cmd
import sys
import argparse
from DGA.File_IO import write_model_args_to_file, load_model_args_from_file, write_error_log
from DGA.Algorithm import Genetic_Algorithm_Base as Algorithm
from DGA.Model import Model
from DGA.Server import Server


class Server_SLURM(Server):
  def __init__(self, run_name: str, algorithm: Algorithm | type, model: Model | type,
               num_parallel_processes: int, sbatch_script: str, call_type: str = 'init',
               data_path: str = None, log_pool: int = -1, **kwargs):
    self.sbatch_script = sbatch_script
    # if call_type == 'init':
    #   self.sbatch_script = os.path.abspath(__file__)  # Need abs path for future calls (original call dir won't be on sys.path anymore)
    super().__init__(run_name, algorithm, model, num_parallel_processes, call_type, data_path, log_pool, **kwargs)

  # Save args for model, and make sbatch calls
  def make_call(self,
                model_id: int,
                gene_name: str,
                call_type: str,
                **kwargs):

    write_model_args_to_file(model_id=model_id,
                              gene_name=gene_name,
                              call_type=call_type,  # callback or run_model
                              run_name=self.run_name,
                              algorithm_path=self.algorithm_path,
                              algorithm_name=self.algorithm_name,
                              model_path=self.model_path,
                              model_name=self.model_name,
                              model_args=self.model_args,
                              num_parallel_processes=self.num_parallel_processes,
                              data_path=self.data_path,
                              log_pool=self.log_pool,
                              sbatch_script=self.sbatch_script,   # Needed for SLURM
                              **kwargs)

    # Call sbatch script
    if call_type == 'run_model':
      server_path_ = os.path.abspath(__file__)  # Get absolute path to current location on machine
      print(f"sbatch {self.sbatch_script} --model_id={model_id} --run_name={self.run_name} --server_path={server_path_}")
      cmd(f"sbatch {self.sbatch_script} --model_id={model_id} --run_name={self.run_name}")
    elif call_type == 'run_server':     # If true, means already on node, no need to make new node
      alg_module_name = self.algorithm_path_.split('/')[-1][:-3]
      alg = getattr(__import__(alg_module_name, fromlist=[alg_module_name]), algorithm_name_)
      self.server_callback(alg, **kwargs)


# Main function catches server-callbacks & runs models
if __name__ == '__main__':
  parser_ = argparse.ArgumentParser()
  parser_.add_argument('--model_id', type=int)
  parser_.add_argument('--run_name', type=str)
  args_ = parser_.parse_args()

  # Load args from file
  all_args = load_model_args_from_file(args_.model_id, args_.run_name)

  # Establish location of Algorithm and Model classes & add them to python path
  algorithm_path_ = all_args['algorithm_path']
  algorithm_name_ = all_args['algorithm_name']
  model_path_ = all_args['model_path']
  model_name_ = all_args['model_name']
  server_path_ = os.path.abspath(__file__)  # Get absolute path to current location on machine
  base_path_ = '/'.join(server_path_.split('/')[0:-2])    # Get path to "./Distributed_GA" ie. base folder
  alg_module_path_ = file_path(base_path_, '/'.join(algorithm_path_.split('/')[0:-1]))
  model_module_path_ = file_path(base_path_, '/'.join(model_path_.split('/')[0:-1]))
  sys.path.append(alg_module_path_)
  sys.path.append(model_module_path_)

  # Create Algorithm and Model objects
  alg_module_name = algorithm_path_.split('/')[-1][:-3]
  model_module_name = model_path_.split('/')[-1][:-3]
  algorithm_ = getattr(__import__(alg_module_name, fromlist=[alg_module_name]), algorithm_name_)
  model_ = getattr(__import__(model_module_name, fromlist=[model_module_name]), model_name_)
  all_args['algorithm'] = algorithm_
  all_args['model'] = model_

  # Run server protocol with bash kwargs
  try:
    Server(**all_args)
  except Exception as e:
    write_error_log(all_args['run_name'], all_args)
    raise e