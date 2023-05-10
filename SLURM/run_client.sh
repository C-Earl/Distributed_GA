#!/bin/bash
echo "Starting client..."
echo "$@"
python3 popen_test.py "$@" --call_type='run_client'
