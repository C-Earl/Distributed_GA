from filelock import FileLock
from os.path import join as file_path
import time

with FileLock(file_path('test_dir') + ".lock"):
  print("hi")
  time.sleep(5)