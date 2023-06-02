import datetime
from GA import Server, Client, Model, Algorithm
import numpy as np
import time
import pickle
from os.path import join as file_path

# Create unique directory based on current time
# now = datetime.datetime.now()
# name = "Test_GA" + now.strftime("_%m_%d-%H_%M_%S")
# server = Server(name, client_name="MyClient.py", num_clients=1,
#                 num_genes=3, gene_shape=10, mutation_rate=0.2, model_name="ASDASd", count=0)

def imitate_server_callback():
  kwargs = {"num_genes":5, "count":0,
            "gene_shape":10, "mutation_rate":0.2}
  model = Model()
  client = Client("Test_GA_06_01-13_03_46", "gene_3edd0f022b61690612094f6b3693353ebac5860e0945b164c93da6338e40fa86", model, **kwargs)
  client.run(**kwargs)

imitate_server_callback()

# def imitate_create_gene():
#   gname = 'gene_e91172adaf7ccc488f5fa01966ed611f7e49ce45e3369df5f02a352e2078f51d'
#   alg = Algorithm(run_name="Test_GA_06_01-12_51_08", gene_shape=10, mutation_rate=0.2,num_genes=3, recall=True,
#                   fitness=123, gene_name=gname) # kwargs
#   gene = alg.create_gene()
#
# imitate_create_gene()