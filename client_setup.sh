#!/bin/bash
echo "Starting client $1 (local?: $2)..."
python3 Slurm.py -server=0 -initial_run=0 -id=$1 -local=$2
