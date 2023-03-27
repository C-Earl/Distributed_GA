import pickle
from concurrent.futures import ThreadPoolExecutor

# gRPC
import grpc
import Server_pb2
import Server_pb2_grpc

# Local imports
from Models import Model
import EA


# Server-side gRPC method class
# Remote methods for clients to access EA functions on server
class EAServicer(Server_pb2_grpc.EAServicer):

  def __init__(self, algorithm: EA):
    self.algorithm = algorithm    # Training method


  def get_genes(self, request, context):
    print("request for genes...")
    genes = self.algorithm.generate_new_gene()       # Get genes from EA
    gene_bytes = pickle.dumps(genes)                  # Binarize
    return self.create_return_msg("get_genes", genes=gene_bytes)

  def return_results(self, request, context):
    print("request to return fitness...")
    fitness = request.fitness
    genes = pickle.loads(request.genes)
    success = self.algorithm.add_fitness(genes, fitness)
    return self.create_return_msg("return_results", success=success)

  # Helper function for network messages
  def create_return_msg(self, rpc_func_name: str, **kwargs):
    rpc_func = getattr(Server_pb2, rpc_func_name + "_REP")
    return rpc_func(**kwargs)


class Server(object):
  def __init__(self,
    num_clients: int,  # Number of Clients
    max_threads: int,  # Max number of threads
    service: EAServicer
  ):
    self.num_clients = num_clients
    self.max_threads = max_threads
    self.server = grpc.server(ThreadPoolExecutor(max_workers=max_threads))
    self.service = service

  # Open server for clients to connect to
  def run(self):
    Server_pb2_grpc.add_EAServicer_to_server(self.service, self.server)
    self.server.add_insecure_port('[::]:' + str(port))
    self.server.start()
    print(f"Listening on port {port}")
    self.server.wait_for_termination()


if __name__ == '__main__':
  import argparse

  # CMD Arguments
  parser = argparse.ArgumentParser(prog=f"{__file__}")
  parser.add_argument('-num_clients', type=int)
  parser.add_argument('-max_threads', type=int)
  parser.add_argument('-port', type=int)
  args = parser.parse_args()
  num_clients = args.num_clients
  max_threads = args.max_threads
  port = args.port

  # Define 4 essential components for run: model, algorithm, client, & service
  model = Model(param_file_path="../params.yaml")
  alg = EA.EA(10, model)
  service = EAServicer(algorithm=alg)

  # Compile & run server
  server = Server(5, 5, service)
  server.run()
