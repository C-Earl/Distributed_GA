#!/bin/bash
echo "Server callback..."
echo "$@"
python3 popen_test.py "$@" --call_type='server_callback'
