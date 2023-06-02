import time
import ast
from os import mkdir, rmdir
from os.path import join as file_path, isdir
from filelock import FileLock
import os
import argparse
import pickle
import numpy as np
import sys
import subprocess
import hashlib

# Hash consistently across runs (used for gene file names)
def consistent_hasher(x):
	# hasher = hashlib.sha256()
	# hasher.update(x.encode('utf-8'))
	b = bytes(str(x), 'utf-8')
	return hashlib.sha256(b).hexdigest()  # Get the hexadecimal representation of the hash value

###################### GLOBALS ######################
POOL_DIR = "pool"
LOCK_DIR = "locks"
CLIENT_RUNNER = "run_client.sh"
SERVER_CALLBACK = "run_server.sh"
# GENE_NAME = lambda cid: f"gene_{cid}"   # cid = client id
GENE_NAME = lambda gene_key: f"gene_{consistent_hasher(gene_key)}"  # Hash gene for file name
###################### GLOBALS ######################


# Anything done here should require a lock
################# ASYNC FUNCTIONS ##################

# # Give lock to gene file
# def lock_gene(name: str, run_name: str):
#   lock_path = file_path(run_name, LOCK_DIR, name) + ".lock"
#   if os.path.exists(lock_path):
#     return False
#   else:
#     open(lock_path, 'w').close()  # Create lock file
#     return True


# # Remove lock from gene file
# def unlock_gene(name: str, run_name: str):
#   lock_path = file_path(run_name, LOCK_DIR, name) + ".lock"
#   if os.path.exists(lock_path):
#     os.remove(lock_path)


# Write gene to file
def write_gene(gene: dict, name: str, run_name: str):
	with FileLock(file_path(run_name, LOCK_DIR, name) + ".lock"):
		pool_path = file_path(run_name, POOL_DIR)
		gene_path = file_path(pool_path, name) + ".pkl"
		with open(gene_path, 'wb') as gene_file:
			pickle.dump(gene, gene_file)

	# If exists, remove lock
	os.remove(file_path(run_name, LOCK_DIR, name) + ".lock")
	

# Load gene from file
def load_gene(name: str, run_name: str):
	with FileLock(file_path(run_name, LOCK_DIR, name) + ".lock"):
		pool_path = file_path(run_name, POOL_DIR)
		gene_path = file_path(pool_path, name) + ".pkl"
		with open(gene_path, 'rb') as gene_file:
			gene = pickle.load(gene_file)

	# If exists, remove lock
	os.remove(file_path(run_name, LOCK_DIR, name) + ".lock")
	return gene

# Delete gene file
def delete_gene(name: str, run_name: str):
	with FileLock(file_path(run_name, LOCK_DIR, name) + ".lock"):
		pool_path = file_path(run_name, POOL_DIR)
		gene_path = file_path(pool_path, name) + ".pkl"
		os.remove(gene_path)
	
	# If exists, remove lock
	os.remove(file_path(run_name, LOCK_DIR, name) + ".lock")

def load_pool(run_name: str):
	pool = {}
	pool_path = file_path(run_name, POOL_DIR)
	for root, dirs, files in os.walk(pool_path):
		for file in files:
			gene = load_gene(file.split('.')[0], run_name)
			gene_key = get_pool_key(gene['gene'])
			pool[gene_key] = gene
	return pool

##################################################################


# Recursive. Transform np array into nested tuple (for hashing)
# Assumed all genes are unique
def get_pool_key(gene: np.array):
	if gene.ndim == 0:  # Scalar value
		return gene.item()
	else:
		return tuple(get_pool_key(sub_arr) for sub_arr in gene)

# Recursive. Transform nested tuple (pool key) into np array
# Assumed valid shape for np array
def get_gene(pool_key: tuple):
	# return np.asarray(pool_key)
	if isinstance(pool_key, tuple):
		return np.array([get_gene(sub_tup) for sub_tup in pool_key])
	else:
		return pool_key

# Normalize values to positive range [0, +inf) (fitnesses)
# Do nothing if already in range [0, +inf)
def pos_normalize(values):
	min_v = min(values)
	if min_v < 0:
		return [i + abs(min_v) for i in values]
	else:
		return values


# TODO: Delete this stuff
def test_marker(id, content):
	with open(f"{id}.txt", 'w') as file:
		file.write(str(content))

def client_test_marker(cid, count, gene):
	with open(f"{cid}.txt", 'w') as file:
		file.write(str(count))
		file.write(str(gene))

# - Must comply with Algorithms gene structure
class Model():
	def run(self, gene) -> float:
		# Evaluate gene
		fitness = sum([-(i**2) for i in gene['gene']])
		return fitness


class Server():
	def __init__(self, run_name: str, model_name: str, num_clients: int, recall: bool = False, **kwargs):
		self.run_name = run_name

		### RECALL HANDLING ###
		if recall:
			### Handle kwarg dtypes here ###
			kwargs['num_genes'] = int(kwargs['num_genes'])
			kwargs['mutation_rate'] = float(kwargs['mutation_rate'])

			# TODO: Testing code, delete
			gene_record = ast.literal_eval((kwargs['gene_record']))
			gene_record.append(kwargs['gene_name'][5:10])
			kwargs['gene_record'] = gene_record
			count = int(kwargs['count'])
			time.sleep(1)
			kwargs['count'] = count+1
			if count > 10:
				test_marker("ENDED", kwargs['gene_record'])
				return

			# Call Algorithm to write fitness, get new gene
			alg = Algorithm(run_name, recall=recall, num_clients=num_clients, **kwargs)
			gene = alg.create_gene()
			gene_name = GENE_NAME(gene['name'])
			write_gene(gene, gene_name, run_name)
			kwargs.pop('fitness')   # TODO: Figure out more elegant solution

			# Call client to run new gene
			kwargs.pop('gene_name')     # Remove old gene name
			self.run_client(gene_name=gene_name, **kwargs)
			return

		# Create base & lock directory
		mkdir(self.run_name)
		mkdir(file_path(self.run_name, LOCK_DIR))

		# Start GA algorithm
		self.alg = Algorithm(self.run_name, **kwargs)

		# Run clients on new genes
		pool = self.alg.pool
		new_gene_keys = pool.keys()
		# new_genes = [get_gene(k) for k in new_gene_keys]
		for key in new_gene_keys:
			self.run_client(gene_name=GENE_NAME(key), **kwargs)

	# Recall function
	def run_client(self, gene_name: str, **kwargs):
		### Necessary kwargs for client run ###
		kwargs['gene_name'] = gene_name
		kwargs['run_name'] = self.run_name
		kwargs['call_type'] = "run_client"

		# Convert args/kwargs to bash
		bash_args = []
		for k,v in kwargs.items():
			bash_args.append(f"--{k}={v}")

		# Call client through terminal
		subprocess.Popen(["python3", "GA.py"] + bash_args)
		# bash_args = ' '.join(bash_args)
		# os.system(f"python3 GA.py {' '.join([str(i) for i in bash_args])}")


# - Handles all gene creation and structure
class Algorithm():
	def __init__(self, run_name: str, gene_shape: tuple, mutation_rate: float, num_genes: int = 10, recall: bool = False, **kwargs):
		self.run_name = run_name
		self.gene_shape = gene_shape        # TODO: IMPLEMENT THIS
		self.mutation_rate = mutation_rate  # TODO: IMPLEMENT THIS
		self.num_genes = num_genes
		self.pool_path = file_path(self.run_name, POOL_DIR)
		self.pool = {}

		### RECALL HANDLING ###
		if recall:
			fitness = kwargs.pop('fitness')   # *Always* pop fitness here
			tested_gene_name = kwargs.pop('gene_name')
			fitness = float(fitness)

			# Update pool in files
			try:
				gene_data = load_gene(tested_gene_name, run_name)
			except FileNotFoundError:
				raise Exception(kwargs)
			gene_data['fitness'] = fitness
			gene_data['status'] = 'tested'
			write_gene(gene_data, tested_gene_name, run_name)

			# Load gene pool w/ tested gene
			self.pool = load_pool(run_name)
			gene_data['fitness'] = fitness
			tested_gene_key = get_pool_key(gene_data['gene'])
			self.pool[tested_gene_key] = gene_data

			# # Update pool in files
			# updated_gene = {'gene': get_gene(tested_gene_key), 'name': tested_gene_key,
			# 								'fitness': fitness, 'status': 'tested'}
			# write_gene(updated_gene, tested_gene_name, run_name)

			return

		# Generate pool & files
		mkdir(self.pool_path)
		for i in range(num_genes):
			gene = self.create_gene()
			gene_key = get_pool_key(gene['gene'])
			self.pool[gene_key] = gene
			write_gene(gene, GENE_NAME(gene['name']), run_name)   # Write file

	# Gene structure:
	# {
	#   'gene' : size 10 array of floats,
	#   'name' : any hashable type, unique identifier given to gene
	#   'fitness' : float, fitness of gene (None means untested)
	#   'status' : string, 'being tested', or 'tested'
	# }
	# TODO: Add kwargs
	def create_gene(self):

		# If pool uninitialized, add new untested gene
		if len(self.pool) < self.num_genes:
			gene = np.random.rand(10)
			return {'gene': gene, 'name': get_pool_key(gene), 'fitness': None, 'status': 'being tested'}

		# Trim pool of weakest gene if full
		elif len(self.pool) >= self.num_genes:
			# if None in [kv[1]['fitness'] for kv in self.pool.items()]:
			#   raise Exception(f"{self.pool.items()}")
			valid_parents = {gene_key: gene_data for gene_key, gene_data in self.pool.items()  # Filter genes with no fitness
											 if (not gene_data['status'] == 'being tested')}
			try:
				min_key = min(valid_parents.items(), key=lambda kv: kv[1]['fitness'])[0]
			except ValueError:
				raise Exception(f"{self.pool.items()}")
			gene_data = self.pool[min_key]
			del self.pool[min_key]    # Remove from local pool
			delete_gene(GENE_NAME(gene_data['name']), self.run_name)      # Remove from pool in files

		# If untested gene, have it tested first
		# TODO: Move to new server-call function called "fetch_gene"
		untested_gene_key = self.untested_genes()
		if untested_gene_key:
			return {'gene': get_gene(untested_gene_key), 'name': untested_gene_key,
							'fitness': self.pool[untested_gene_key]['fitness'], 'status': 'being tested'}

		# Select parents for reproduction
		valid_parents = {gene_key : gene_data for gene_key, gene_data in self.pool.items()  # Filter genes with no fitness
										 if (not gene_data['status'] == 'being tested')}
		sorted_parents = sorted(valid_parents.items(), key=lambda gene_kv: gene_kv[1]['fitness'])   # Sort by fitness
		sorted_parents = [get_gene(gene_key) for gene_key, _ in sorted_parents]     # Transform gene_keys to genes
		fitness_scores = [gene_data['fitness'] for _, gene_data in valid_parents.items()]   # Get fitness's (unordered)
		normed_fitness = pos_normalize(fitness_scores)            # Shift fitness's to [0, +inf)
		probabilities = normed_fitness / np.sum(normed_fitness)   # Normalize to [0, 1]
		p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)
		p1, p2 = sorted_parents[p1_i], sorted_parents[p2_i]       # Weighted selection based on fitness

		# Generate offspring with crossover
		crossover_point = np.random.randint(0, self.gene_shape)
		child = np.concatenate((p1[:crossover_point], p2[crossover_point:]))

		# Random mutation
		if np.random.rand() < 0.1:
			mutation_point = np.random.randint(0, self.gene_shape)
			child[mutation_point] += np.random.uniform(-self.mutation_rate, +self.mutation_rate)

		return {'gene': child, 'name': get_pool_key(child), 'fitness': None, 'status': 'being tested'}

	# Return untested gene if there is one in pool
	# False otherwise
	def untested_genes(self):
		for gene_key, gene_data in self.pool.items():
			if gene_data['status'] == 'being tested':
				return gene_key
		return False


class Client():
	def __init__(self, run_name: str, gene_name: str, model, **kwargs):
		self.run_name = run_name
		self.gene_name = gene_name
		self.model = model
		self.gene = load_gene(gene_name, run_name)

	def run(self, **kwargs):
		# Run model
		fitness = self.model.run(self.gene)

		# Write fitness (attached to gene)
		# self.gene['fitness'] = fitness
		# write_gene(self.gene, self.gene_name, self.run_name)

		# Initiate callback
		self.callback(fitness, **kwargs)

	def callback(self, fitness: int, **kwargs):
		# Convert args/kwargs to bash
		kwargs['gene_name'] = self.gene_name
		kwargs['run_name'] = self.run_name
		kwargs['fitness'] = fitness
		kwargs['call_type'] = "server_callback"

		# Convert args/kwargs to bash
		bash_args = []
		for k, v in kwargs.items():
			bash_args.append(f"--{k}={v}")

		# Callback server through terminal
		subprocess.Popen(["python3", "GA.py"] + bash_args)
		# bash_args = ' '.join(bash_args)
		# os.system(f"python3 GA.py {' '.join([str(i) for i in bash_args])}")


if __name__ == '__main__':
	# Parse unknown num. of arguments. *All strings*
	parser = argparse.ArgumentParser()
	for arg in sys.argv[1:]:    # https://stackoverflow.com/questions/76144372/dynamic-argument-parser-for-python
		if arg.startswith('--'):
			parser.add_argument(arg.split('=')[0])
	all_args = vars(parser.parse_args())
	call_type = all_args.pop('call_type')

	# Handle server calling client
	if call_type == "run_client":
		# Pop important params
		kwargs_ = all_args
		gene_name_ = kwargs_.pop('gene_name')    # '_' for avoiding global def conflict
		run_name_ = kwargs_.pop('run_name')

		# Run client
		model = Model()   # TODO: Figure out how to pass models
		client = Client(run_name_, gene_name_, model, **kwargs_)
		client.run(**kwargs_)

	# Handle client callback to server
	elif call_type == "server_callback":
		# Pop important params
		kwargs = all_args
		run_name_ = kwargs.pop('run_name')
		server = Server(run_name_, model_name='placeholder', num_clients=1, recall=True, **kwargs) # TODO: Client num

	else:
		print(f"error, improper call_type: {call_type}")


		# ordered_genes = sorted(self.pool.items(), key=lambda x: x['fitness'], reverse=True)
		# p1 = ordered_genes[0]['gene']
		# p2 = ordered_genes[1]['gene']
		# crossover_point = np.random.randint(1, len(p1) - 1)
		# offspring_gene = np.concatenate((p1[:crossover_point], p2[crossover_point:]))