from DGA.Algorithm import Genetic_Algorithm
from DGA.Client import Client
from DGA.Server import Server
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from math import prod

DEVICE = torch.device('cpu')    # CPU will be faster than GPU for this example

class ArtificialNeuralNet(nn.Module):
  def __init__(self, gene):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 120)
    self.fc2 = nn.Linear(120, 60)
    self.fc3 = nn.Linear(60, 10)

    # Set weights manually based on gene
    weights = [gene['l1'], gene['l2'], gene['l3']]
    with torch.no_grad():
      self.fc1.weight = torch.nn.parameter.Parameter(torch.tensor(weights[0], dtype=torch.float32).T.to(DEVICE))
      self.fc2.weight = torch.nn.parameter.Parameter(torch.tensor(weights[1], dtype=torch.float32).T.to(DEVICE))
      self.fc3.weight = torch.nn.parameter.Parameter(torch.tensor(weights[2], dtype=torch.float32).T.to(DEVICE))

  def forward(self, x):
    x = x.to(DEVICE)
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


# # # # # # # # # # # # # # # # # # # # # #
# Use the Client class to load your own models into DGA. the run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
class Complex_GA_Client(Client):  # <--- Remember to inherit Client class
  def load_data(self, **kwargs):
    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=ToTensor(),
                                          download=True)
    self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=ToTensor(),
                                         download=True)
    self.testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                             shuffle=False, num_workers=2)

  def run(self, gene, **kwargs) -> float:

    # Create optimizer and ANN
    ANN = ArtificialNeuralNet(gene)
    ANN.to(DEVICE)

    # Test ANN
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
      for data in self.testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = ANN(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Return fitness (ie. accuracy)
    return correct / total


class Complex_GA(Genetic_Algorithm):  # <--- Remember to inherit Genetic_Algorithm class

  # Initialize with random values
  # Called automatically if pool is not full
  def initial_gene(self, **kwargs):
    return {
      'l1': np.random.rand(28 * 28, 120),
      'l2': np.random.rand(120, 60),
      'l3': np.random.rand(60, 10)
    }

  # Remove worst gene from pool
  # TODO: Note it doesn't use actual gene data
  def remove_weak(self, gene_pool: dict):
    sorted_parents = sorted(gene_pool.items(),
                            key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)  # Sort by fitness
    worst_gene = sorted_parents[-1][0]
    del gene_pool[worst_gene]  # Remove from pool obj
    return gene_pool

  # Weighted selection of parents based on fitness
  # TODO: Note it doesn't use actual gene data
  def select_parents(self, gene_pool: dict):
    fitness_scores = [gene_data['fitness'] for _, gene_data in gene_pool.items()]  # Get fitness's (unordered)
    normed_fitness = self.pos_normalize(fitness_scores)  # Shift fitness's to [0, +inf)
    probabilities = normed_fitness / np.sum(normed_fitness)  # Normalize to [0, 1]
    p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)
    sorted_genes = sorted(gene_pool.items(), key=lambda gene_kv: gene_kv[1]['fitness'], reverse=True)
    return sorted_genes[p1_i][1]['gene'], sorted_genes[p2_i][1]['gene'] # TODO: make this simpler

  # Crossover parents at random point
  def crossover(self, p1, p2):
    new_gene = {
      'l1': None,
      'l2': None,
      'l3': None
    }
    for lname, layer in p1.items():
      crossover_point = np.random.randint(0, prod(self.gene_shape[lname]))
      new_gene[lname] = np.concatenate((p1[lname][:crossover_point], p2[lname][crossover_point:]))
    return new_gene

  # Mutate gene at random point
  def mutate(self, gene):
    for lname, layer in gene.items():
      if np.random.rand() < 0.0:
        mutation_point = np.random.randint(0, self.gene_shape[0])
        gene[mutation_point] += np.random.uniform(-self.mutation_rate, +self.mutation_rate)
    return gene

  # Normalize values to positive range [0, +inf) (fitnesses)
  # Do nothing if already in range [0, +inf)
  def pos_normalize(self, values):
    min_v = min(values)
    if min_v < 0:
      return [i + abs(min_v) for i in values]
    else:
      return values


if __name__ == '__main__':
  import os

  alg_path = os.path.abspath(__file__)
  client_path = os.path.abspath(__file__)
  gene_shapes = {'l1': (28 * 28, 120), 'l2': (120, 60), 'l3': (60, 10)}
  Server(run_name="complex_example",
         algorithm=Complex_GA,
         client=Complex_GA_Client,
         num_parallel_processes=2, gene_shape=gene_shapes, num_genes=10, mutation_rate=0.1,
         iterations=3,)
