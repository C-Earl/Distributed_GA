from DGA.Plotting import plot_model_logs

if __name__ == '__main__':
  import sys
  print(sys.path)
  plot_model_logs(run_dir="./scripts/Walker2D/Walker_2D_run", num_models=1)