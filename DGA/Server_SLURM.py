import logging
import os.path
import subprocess
from os.path import join as file_path
from os import system as cmd
import sys
import argparse
from portalocker.exceptions import LockException
from DGA.File_IO import write_model_args_to_file, load_model_args_from_file, write_error_log, load_agent_job_ID, \
  save_agent_job_ID, RUN_INFO_DIR, JOB_ID_DIR, LOG_DIR, SERVER_LOG_DIR
from DGA.Algorithm import Genetic_Algorithm_Base as Algorithm
from DGA.Model import Model
from DGA.Server import Server


class Server_SLURM(Server):
  def __init__(self, run_name: str, algorithm: Algorithm | type, model: Model | type,
               num_parallel_processes: int, sbatch_script: str, callback_sbatch_script: str = None, call_type: str = 'init',
               data_path: str = None, log_pool: int = -1, **kwargs):
    self.sbatch_script = sbatch_script
    # self.callback_sbatch_script = callback_sbatch_script

    # Create job_id directory. Keep track of what agent is running what job
    os.makedirs(file_path(run_name, RUN_INFO_DIR, JOB_ID_DIR), exist_ok=True)

    super().__init__(run_name, algorithm, model, num_parallel_processes, call_type, data_path, log_pool, **kwargs)

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
      out_string = subprocess.check_output(f"sbatch {self.sbatch_script} {agent_id} {self.run_name} {server_path_}")
      job_id = str(int(out_string.split()[-1]))        # out_string = "Submitted batch job <job-id>"
      save_agent_job_ID(self.run_name, agent_id, job_id)
    elif call_type == 'server_callback':     # If true, means already on node, no need to make new node
      self.server_callback(**kwargs, agent_id=agent_id, params_name=params_name)

  # Overide for Server.server_callback
  # Check if 'neighbor' agent is still running
  def server_callback(self, agent_id: int, params_name: str, **kwargs):
    # Load runtime for agent (self.agent_id + 1)
    # If last agent, load agent 0
    neighbor_agent_id = (agent_id + 1) % self.num_parallel_processes
    neighbor_agent_job_id = load_agent_job_ID(self.run_name, neighbor_agent_id)

    # Check neighbor still running
    seff_command = ["seff", neighbor_agent_job_id]
    completed_process = subprocess.run(seff_command, capture_output=True)
    run_state = str(completed_process.stdout).split('State: ')[1].split('\\nNodes')[0]
    # acceptable_states = ['PENDING', 'RUNNING', 'COMPLETED', 'COMPLETING', "SUSPENDED", "CONFIGURING"]
    crash_states = ['', 'BOOT_FAIL', 'DEADLINE', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'TIMEOUT']
    # agent_job_file_exists = os.path.exists(file_path(self.run_name, RUN_INFO_DIR, JOB_ID_DIR, f"AGENT_{neighbor_agent_id}_job_id"))

    print(f"DEBUG: {run_state=}, {neighbor_agent_job_id=}, {agent_id=}")

    # Cases:
    # 1) If seff says crashed, then it crashed
    # 2) If seff says it's completed, but agent_job_id file is not deleted, then it crashed

    # If seff says crashed
    restart_agent = False
    if run_state in crash_states:
      restart_agent = True

    # If seff says completed, but agent_job_id file wasn't deleted
    elif run_state == 'COMPLETED (exit code 0)':
      restart_agent = True

    # If node was preempted by SLURM
    elif run_state == 'PREEMPTED (exit code 0)':
      restart_agent = True

    if restart_agent:
      logging.debug(f"Agent {neighbor_agent_id} crashed on job {neighbor_agent_job_id}. Restarting...")

      # Find neighbors assigned params
      neighbor_args = load_model_args_from_file(neighbor_agent_id, self.run_name)
      neighbor_params_name = neighbor_args['params_name']
      self.make_call(neighbor_agent_id, neighbor_params_name, 'run_model', **kwargs)
    else:
      logging.debug(f"Agent {neighbor_agent_id} is still running on job {neighbor_agent_job_id}. Proceeding...")


    # Proceed with normal callback
    super().server_callback(**kwargs, agent_id=agent_id, params_name=params_name)


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

  # Create logger for server
  logging.basicConfig(filename=str(file_path(all_args['run_name'], LOG_DIR, SERVER_LOG_DIR, f"AGENT_{args_.agent_id}.log")),
                      encoding='utf-8',
                      level=logging.DEBUG)

  # Run server protocol with bash kwargs
  try:
    Server_SLURM(**all_args)
  except Exception as e:
    logging.exception(f" Error on Agent {args_.agent_id}:")
    raise

# import numpy as np
# job_id = str(np.random.randint(1000000))
# cmd(f"sbatch {self.sbatch_script} {agent_id} {self.run_name} {server_path_}")
# print(f"Submitted job {job_id} for agent {agent_id}")