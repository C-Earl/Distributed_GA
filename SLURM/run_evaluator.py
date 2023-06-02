import os.path
import pickle
from os.path import join as file_path
RUN_NAME = "Test_GA_06_01-13_03_46"
POOL_DIR = "pool"

# Load gene history
pool_path = file_path(RUN_NAME, POOL_DIR)
gene_creation_times = {}
for root, dirs, files in os.walk(pool_path):
  for file in files:
    gene_path = file_path(pool_path, file)
    gene_creation_times[file] = os.path.getmtime(gene_path)

# Sort genes by creation date
gene_progression = sorted(gene_creation_times.items(), key=lambda kv: kv[1])

# Get fitnesses
fitnesses = []
genes = []
for gene in gene_progression:
  gene_name = gene[0]
  gene_path = file_path(pool_path, gene_name)
  with open(gene_path, 'rb') as gene_file:
    gene_data = pickle.load(gene_file)
    genes.append(gene_data['gene'])
    fitnesses.append(gene_data['fitness'])

print("hji")
