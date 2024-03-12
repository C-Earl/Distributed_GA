from DGA.Server import Server
from DGA.Algorithm import Genetic_Algorithm
from DGA.Local import Synchronized
from Walker_Model import Walker_Model
from DGA.Gene import Gene, Genome

# NOTE: Need to make run file SEPARATE from the file containing any custom objects.
#       Specifically, don't start a run *anywhere* in the same file as the custom objects
#       This is because: To reload custom objects from file with pickle, the file must
if __name__ == '__main__':
  # Run variables
  VECTOR_SHAPE = (10, 10)

  # Genome
  genome = Genome()
  genome.add_gene(Gene(dtype=float,
                       min_val=0.01, max_val=3,
                       mutate_rate=0.2, mutate_scale=0.02), 'expl_noise')
  genome.add_gene(Gene(dtype=float,
                       min_val=0.9, max_val=0.999,
                       mutate_rate=0.2, mutate_scale=0.005), 'discount')
  genome.add_gene(Gene(dtype=float,
                       min_val=0.0010, max_val=0.0100,
                       mutate_rate=0.2, mutate_scale=0.0002), 'tau')
  genome.add_gene(Gene(dtype=float,
                       min_val=0.05, max_val=0.5,
                       mutate_rate=0.2, mutate_scale=0.02), 'policy_noise')
  genome.add_gene(Gene(dtype=float,
                       min_val=0.1, max_val=1.0,
                       mutate_rate=0.2, mutate_scale=0.02), 'noise_clip')

  mod = Walker_Model()
  alg = Genetic_Algorithm(genome=genome, pool_size=10, iterations=1_000)
  sync_runner = Synchronized(run_name="Walker_2D_run", algorithm=alg, model=mod, log_freq=1)
  sync_runner.run()

  from DGA.Plotting import plot_model_logs
  plot_model_logs(run_dir="Walker_2D_run", num_models=1)