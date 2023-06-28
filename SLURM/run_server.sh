#!/bin/bash
#SBATCH --job-name=BN_GA_S
#SBATCH --time=00-24:30:00
#SBATCH --mem=8192
#SBATCH --cpus-per-task=2
#SBATCH --signal=B:USR1@30
#SBATCH --output=logs_gpu/%x.%j.out
##SBATCH --error=error_gpu/R-%x.%j.err
#SBATCH -p preempt
#SBATCH --exclude=c1cmp[025-026],p1cmp[075,090-109]
#SBATCH --gres=gpu:1

# trap at https://hpc-discourse.usc.edu/t/signalling-a-job-before-time-limit-is-reached/314/3



### Environment Setup ###
# Activate conda env
source /cluster/tufts/levinlab/hhazan01/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate delayW
unset DISPLAY
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/tufts/levinlab/hhazan01/miniconda3/lib/

# load Cuda 11
module load cuda/11.0
echo "-------------1---------------"
echo $SLURMD_NODENAME
echo "-------------2---------------"
nvidia-smi
echo "-------------3---------------"
printenv CUDA_VISIBLE_DEVICES
echo "-------------4---------------"

trap 'echo signal recieved!; kill "${PID}"; wait "${PID}"; handler' USR1 SIGINT SIGTERM


### Run Model Script (run_client) ###
# Will lead to server callback & re-queuing of node
python3 "Server_SLURM.py" "$@" &    # Run server script with all runtime args
#echo "Job started!"
#PID="$!"
#wait "${PID}"
#echo "Job ended!"