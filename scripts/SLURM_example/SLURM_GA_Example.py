from DGA.Algorithm import Genetic_Algorithm
from DGA.Model import Model
from DGA.Server_SLURM import Server_SLURM

# Same as example from GA_examples/Simple_GA_Example.py
# See code from there for more details
class Simple_Model(Model):
  def run(self, gene, **kwargs) -> float:
    fitness = sum([-(i ** 2) for i in gene.flatten()])
    self.gene = gene.tolist()
    return fitness

if __name__ == '__main__':
  alg = Genetic_Algorithm(gene_shape=(100,100),
                          num_genes=25,
                          mutation_rate=0.25,
                          iterations=100,)
  Server_SLURM(run_name="Simple_GA_Example", 
               algorithm=alg,
               sbatch_script="sbatch_script.sh",
               model=Simple_Model(),
               num_parallel_processes=5, )
