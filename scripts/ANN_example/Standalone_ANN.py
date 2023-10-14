# Made using code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=mnist

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch.optim as optim

EPOCHS = 2
DEVICE = torch.device('cpu')    # CPU will be faster than GPU for this example

class ArtificialNeuralNet(nn.Module):
  def __init__(self, gene):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 120).to(DEVICE)
    self.fc2 = nn.Linear(120, 60).to(DEVICE)
    self.fc3 = nn.Linear(60, 10).to(DEVICE)

    # Set weights manually based on gene
    weights = [gene['l1'], gene['l2'], gene['l3']]
    with torch.no_grad():
      self.fc1.weight = torch.nn.parameter.Parameter(torch.tensor(weights[0], dtype=torch.float32).T.to(DEVICE))
      self.fc2.weight = torch.nn.parameter.Parameter(torch.tensor(weights[1], dtype=torch.float32).T.to(DEVICE))
      self.fc3.weight = torch.nn.parameter.Parameter(torch.tensor(weights[2], dtype=torch.float32).T.to(DEVICE))
    # for i, param in enumerate(self.parameters()):
    #   param.data = torch.nn.parameter.Parameter(torch.tensor(weights[i]).to(DEVICE))

  def forward(self, x):
    x = x.to(DEVICE)
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

if __name__ == '__main__':

  # Load data
  trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=ToTensor(),
                                        download=True)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,
                                            shuffle=True, num_workers=2)
  testset = torchvision.datasets.MNIST(root='./data', train=False, transform=ToTensor(),
                                       download=True)
  testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                           shuffle=False, num_workers=2)

  # Fake gene (as an example)
  gene = {
    'l1': np.random.rand(28 * 28, 120),
    'l2': np.random.rand(120, 60),
    'l3': np.random.rand(60, 10)
  }

  # Create optimizer and ANN
  ANN = ArtificialNeuralNet(gene)
  ANN.to(DEVICE)

  # Test trained ANN
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      # calculate outputs by running images through the network
      outputs = ANN(images)
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels.to(DEVICE)).sum().item()

  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')