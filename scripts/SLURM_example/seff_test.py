import subprocess
import time

from DGA.File_IO import save_agent_job_ID, load_agent_job_ID, load_model_args_from_file

if __name__ == '__main__':
  neighbor_agent_job_id = "manually add input here!" # load_agent_job_ID(run_name, neighbor_agent_id)

  # Test sbatch
  out_string = subprocess.check_output(f"sbatch test_sbatch_script.sh")
  job_id = int(out_string.split()[-1])  # out_string = "Submitted batch job <job-id>"

  # Check neighbor init state
  run_state = subprocess.check_output(f"seff {job_id} | grep 'State'")
  run_state = run_state.split()[1]
  print("DEBUG, init_state: ", run_state)

  time.sleep(30)

  # Check neighbor running state
  run_state = subprocess.check_output(f"seff {job_id} | grep 'State'")
  run_state = run_state.split()[1]
  print("DEBUG, run_state: ", run_state)
