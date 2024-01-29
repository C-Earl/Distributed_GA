from DGA.Algorithm import Genetic_Algorithm
from DGA.Model import Testing_Model
from DGA.Gene import Gene, Genome, Parameters
from DGA.Local import Synchronized
from DGA.Server_SLURM import Server_SLURM
from DGA.Server import Server

if __name__ == '__main__':
  # Run variables
  VECTOR_SHAPE = (10, 10)

  # Genome
  genome = Genome()
  gene = Gene(shape=VECTOR_SHAPE, dtype=float, min_val=-10, max_val=10)
  genome.add_gene(gene, 'vector_gene')

  mod = Testing_Model(genome=genome, vector_size=VECTOR_SHAPE, vector_distribution=10, vector_scale=3)
  alg = Genetic_Algorithm(genome=genome, num_params=10, iterations=1_000)
  parallel_runner = Server(run_name="SLURM_run", algorithm=alg, model=mod, num_parallel_processes=4)
  # sync_runner = Synchronized(run_name="SLURM_run", algorithm=alg, model=mod)
  # sync_runner.run()
  #
  # from DGA.Plotting import plot_model_logs
  # plot_model_logs(run_dir="SLURM_run", num_models=1)
