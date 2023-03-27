import pickle
from Models import Model

# gRPC
import grpc
import Server_pb2
from Server_pb2_grpc import EAStub

class Client(object):
  def __init__(
    self,
    server_ip: str,
    model: Model,
  ):
    self.model = model
    self.server_ip = server_ip

  # Client request for new genes from EA
  # Used for Server.get_genes() called in Server.py
  def get_genes(self):
    pass

  # Client request to return fitness for tested genes
  # Used for Server.return_results() called in Server.py
  def return_results(self):
    pass

  def run(self):
    # Get genes from server
    print("Requesting Genes...")
    with grpc.insecure_channel(self.server_ip) as channel:
      stub = EAStub(channel)
      req_msg = Server_pb2.get_genes_REQ(id="0")
      rep_msg = stub.get_genes(req_msg)
      genes = rep_msg.genes

    # Test model fitness
    print("Testing Genes...")
    fitness = self.model.test(genes)

    # Return fitness & get new genes from server
    print("Returning Genes...")
    with grpc.insecure_channel(self.server_ip) as channel:
      stub = EAStub(channel)
      gene_bytes = pickle.dumps(genes)
      req_msg = Server_pb2.return_results_REQ(id="0", genes=gene_bytes, fitness=fitness)
      stub.return_results(req_msg)


if __name__ == '__main__':
  import argparse

  # CMD Arguments
  parser = argparse.ArgumentParser(prog=f"{__file__}")
  parser.add_argument('-server_ip', type=str)
  args = parser.parse_args()
  server_ip = args.server_ip

  model = Model(param_file_path="../params.yaml")
  client = Client(server_ip, model)
  client.run()
