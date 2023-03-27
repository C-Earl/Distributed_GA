#!/bin/bash
echo "Callback for client $1..."
server=1
initial_run=0
python3 Slurm.py -server $server -initial_run $initial_run -id $1