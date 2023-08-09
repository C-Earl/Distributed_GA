from DGA.Algorithm import Genetic_Algorithm_Base, Genetic_Algorithm
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
    self.fc2 = nn.Linear(120, 10)
    # self.fc3 = nn.Linear(60, 10)

    # Set weights manually based on gene
    weights = [gene['l1'], gene['l2']]# , gene['l3']]
    with torch.no_grad():
      self.fc1.weight = torch.nn.parameter.Parameter(torch.tensor(weights[0], dtype=torch.float32).T.to(DEVICE))
      self.fc2.weight = torch.nn.parameter.Parameter(torch.tensor(weights[1], dtype=torch.float32).T.to(DEVICE))
      # self.fc3.weight = torch.nn.parameter.Parameter(torch.tensor(weights[2], dtype=torch.float32).T.to(DEVICE))

  def forward(self, x):
    x = x.to(DEVICE)
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    # x = self.fc3(x)
    return x


if __name__ == '__main__':
  import os

  alg_path = os.path.abspath(__file__)
  client_path = os.path.abspath(__file__)
  gene_shapes = {'l1': (28 * 28, 120), 'l2': (120, 10)}
  Server(run_name="complex_example",
         algorithm=Genetic_Algorithm,
         client=Complex_GA_Client,
         num_parallel_processes=2, gene_shape=gene_shapes, num_genes=4, mutation_rate=1,
         iterations=20,)
