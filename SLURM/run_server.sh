#!/bin/bash
#SBATCH --job-name=Dist_GA
#SBATCH --time=00-24:30:00
#SBATCH --mem=8192
#SBATCH --cpus-per-task=2
##SBATCH --signal=B:USR1@30
#SBATCH --output=logs/%x.%j.out
#SBATCH -p preempt

# trap at https://hpc-discourse.usc.edu/t/signalling-a-job-before-time-limit-is-reached/314/3

### Environment Setup ###
# TODO: Make sure to get requirements for both Server.py and whatever client model needs
# Activate conda env
source /cluster/tufts/levinlab/hhazan01/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate DistribGA
unset DISPLAY
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/tufts/levinlab/hhazan01/miniconda3/lib/


### Run Model Script (run_client) ###
# Will lead to server callback & re-queuing of node
# Note: args expected in order
python3 Server_SLURM.py $1 $2

#echo "Job started!"
#PID="$!"
#wait "${PID}"
#echo "Job ended!"