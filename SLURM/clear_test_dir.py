import os

# Clear logs
for root, dirs, files in os.walk("test_dir/logs"):
  for file in files:
    os.remove(os.path.join(root, file))

# Clear pool
for root, dirs, files in os.walk("test_dir/pool"):
  for file in files:
    os.remove(os.path.join(root, file))

# Clear locks
for root, dirs, files in os.walk("test_dir/run_args"):
  for file in files:
    os.remove(os.path.join(root, file))
