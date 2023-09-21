from matplotlib import pyplot as plt
from os.path import join as file_path
from DGA.pool_functions import read_log

def plot_client_logs(run_dir, num_clients, ax=None):
  for i in range(num_clients):
    log = read_log(run_dir, i)
    x = range(len(log))
    y = [entry['fitness'] for entry in log]
    if ax is None:
      plt.scatter(x, y)
    else:
      ax.scatter(x, y)
  plt.show()