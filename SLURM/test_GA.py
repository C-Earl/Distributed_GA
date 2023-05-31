import datetime
from GA import Server, Client, Model
import numpy as np
import time
import pickle
from os.path import join as file_path

# Create unique directory based on current time
now = datetime.datetime.now()
name = "Test_GA" + now.strftime("_%m_%d-%H_%M_%S")
server = Server(name, client_name="MyClient.py", num_clients=1,
                num_genes=3, gene_shape=10, mutation_rate=0.2, model_name="ASDASd", count=0)

# def imitate_server_callback():
#   fitness = 1234
#   kwargs = {"test":12345, "client_name":"MyClient.py", "num_genes":5, "count":0}
#   model = Model()
#   client = Client("Test_GA_05_31-08_41_27", 0, "gene_0", model, **kwargs)
#   client.run(**kwargs)
#
# imitate_server_callback()