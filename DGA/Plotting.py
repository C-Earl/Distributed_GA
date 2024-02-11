from __future__ import annotations

from matplotlib import pyplot as plt
from os.path import join as file_path
from DGA.File_IO import read_log

def plot_model_logs(run_dir, num_models, ax=None, show_plot=True, save_plot: bool | str = False):
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

  if show_plot:
    plt.show()

  if save_plot:
    plt.savefig(file_path(run_dir, f"fitness_vs_iteration.png"))
    plt.close()
