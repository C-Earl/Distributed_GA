import subprocess

from DGA.File_IO import save_agent_job_ID, load_agent_job_ID, load_model_args_from_file

if __name__ == '__main__':
  neighbor_agent_job_id = "manually add input here!" # load_agent_job_ID(run_name, neighbor_agent_id)

  # Check neighbor still running
  run_state = subprocess.check_output(f"seff {neighbor_agent_job_id} | grep 'State'")
  run_state = run_state.split()[1]
  print("DEBUG, run_state: ", run_state)
