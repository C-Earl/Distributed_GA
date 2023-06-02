import subprocess
import sys
import argparse
import time

from GA import Model, Client, Server
import os
POOL_DIR = "pool"
CLIENT_RUNNER = "run_client.sh"
SERVER_CALLBACK = "run_server.sh"

def test_marker(id):
  with open(f"{id}.txt", 'w') as file:
    file.write("hi")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
    if arg.startswith('--'):  # Add dynamic number of args to parser
      parser.add_argument(arg.split('=')[0])
  all_args = vars(parser.parse_args())
  call_type = all_args.pop('call_type')

  if call_type == "run_client":
    # print("Client")
    count = int(all_args['count'])
    time.sleep(3)
    p = subprocess.Popen(["python3", "popen_test.py", "--call_type=server_callback", f"--count={count+1}"])

  elif call_type == "server_callback":
    # print("Server")
    bash_args = []
    count = int(all_args['count'])
    bash_args.append(f"--count={count+1}")
    bash_args.append(f"--call_type=run_client")
    # test_marker(0)
    time.sleep(3)
    p = subprocess.Popen(["python3", "popen_test.py", "--call_type=run_client", f"--count={count}"])
    # for i in range(10):
    #   time.sleep(1)
    #   print(p.stdout)
    # os.system("bash" + f" ./{CLIENT_RUNNER} ")  # TODO: Popen
  else:
    print(f"error, improper call_type: {call_type}")
