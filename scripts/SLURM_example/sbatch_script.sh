#!/bin/bash
#SBATCH --job-name=Distributed_GA
#SBATCH --time=00-24:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --signal=B:USR1@30
#SBATCH --output=logs_cpu/%x.%j.out
##SBATCH --error=error_cpu/R-%x.%j.err
#SBATCH -p preempt

# trap at https://hpc-discourse.usc.edu/t/signalling-a-job-before-time-limit-is-reached/314/3

# Activate conda env
source /cluster/tufts/levinlab/hhazan01/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate DistribGA
unset DISPLAY
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/tufts/levinlab/hhazan01/miniconda3/lib/

trap 'echo signal recieved!; kill "${PID}"; wait "${PID}"; handler' USR1 SIGINT SIGTERM

echo "Job started!"

# run python script
python3 --model_id $3 --run_name $1 --server_path $2


echo "Job ended!"