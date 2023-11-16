from matplotlib import pyplot as plt
from os.path import join as file_path
from DGA.File_IO import read_log

def plot_model_logs(run_dir, num_models, ax=None):
  for i in range(num_models):
    log = read_log(run_dir, i)
    x = [gene_data['iteration'] for gene_data in log]
    y = [gene_data['fitness'] for gene_data in log]
    if ax is None:
      plt.scatter(x, y, s=1)
    else:
      ax.scatter(x, y, s=1)

  plt.title(f"'{run_dir}' Fitness vs. Iteration")
  plt.xlabel("Iteration")
  plt.ylabel("Fitness")
  plt.show()