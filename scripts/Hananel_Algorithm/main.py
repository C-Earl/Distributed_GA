# from Hananel_Algorithm import Hananel_Algorithm as Algorithm
from DGA.Algorithm import Hananel_Algorithm
from DGA.Server import Server
from DGA.Local import Synchronized
from DGA.Model import Testing_Model as Model
from DGA.Gene import Gene
from Hananel_Genome import Hananel_Genome as Genome

# NOTE: Need to make run file SEPARATE from the file containing any custom objects.
#       Specifically, don't start a run *anywhere* in the same file as the custom objects
#       This is because: To reload custom objects from file with pickle, the file must
if __name__ == '__main__':
  # Run variables
  VECTOR_SHAPE = (10, 10)

  # Genome
  genome = Genome()
  gene = Gene(shape=VECTOR_SHAPE, dtype=float, min_val=-10, max_val=10, mutation_rate=0.9, mutation_scale=1)
  genome.add_gene(gene, 'vector_gene')

  mod = Model(genome=genome, vector_size=VECTOR_SHAPE, vector_distribution=10, vector_scale=3)
  alg = Hananel_Algorithm(genome=genome, num_params=10, iterations_per_epoch=1_000, epochs=2, plateau_warmup=100, mutation_rate=0.9)
  # parallel_runner = Server(run_name="Hananel_Alg", algorithm=alg, model=mod, num_parallel_processes=10)
  sync_runner = Synchronized(run_name="Hananel_Alg", algorithm=alg, model=mod)
  sync_runner.run()

  from DGA.Plotting import plot_model_logs
  plot_model_logs(run_dir="Hananel_Alg", num_models=1)