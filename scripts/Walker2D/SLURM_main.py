from DGA.Algorithm import Genetic_Algorithm
from DGA.Server_SLURM import Server_SLURM
from Walker_Model import Walker_Model
from DGA.Gene import Gene, Genome

if __name__ == '__main__':
  # Run variables
  VECTOR_SHAPE = (10, 10)

  # Genome
  genome = Genome()
  genome.add_gene(Gene(dtype=float,
                       min_val=0.01, max_val=3,
                       mutate_rate=0.2, mutate_scale=0.02), 'expl_noise')
  genome.add_gene(Gene(dtype=float,
                       min_val=0.8, max_val=0.999,
                       mutate_rate=0.2, mutate_scale=0.05), 'discount')
  genome.add_gene(Gene(dtype=float,
                       min_val=0.0010, max_val=0.1000,
                       mutate_rate=0.2, mutate_scale=0.002), 'tau')
  genome.add_gene(Gene(dtype=float,
                       min_val=0.05, max_val=0.5,
                       mutate_rate=0.2, mutate_scale=0.02), 'policy_noise')
  genome.add_gene(Gene(dtype=float,
                       min_val=0.1, max_val=1.0,
                       mutate_rate=0.2, mutate_scale=0.1), 'noise_clip')

  mod = Walker_Model()
  alg = Genetic_Algorithm(genome=genome, pool_size=10, iterations=20_000)
  server = Server_SLURM(run_name="Walker_2D_run", algorithm=alg, model=mod,
                        num_parallel_processes=10, sbatch_script="sbatch_script.sh")


  # sync_runner = Synchronized(run_name="Walker_2D_run", algorithm=alg, model=mod, log_freq=1)
  # sync_runner.run()
  #
  # from DGA.Plotting import plot_model_logs
  # plot_model_logs(run_dir="Walker_2D_run", num_models=1)