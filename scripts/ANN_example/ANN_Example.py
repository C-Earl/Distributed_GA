from DGA.Algorithm import Genetic_Algorithm_Base, Genetic_Algorithm
from DGA.Model import Model
from DGA.Server import Server
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

DEVICE = torch.device('cpu')    # CPU will be faster than GPU for this example


class ArtificialNeuralNet(nn.Module):
  def __init__(self, gene):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 120)
    self.fc2 = nn.Linear(120, 10)

    # Set weights manually based on gene
    weights = [gene['l1'], gene['l2']]
    with torch.no_grad():
      self.fc1.weight = torch.nn.parameter.Parameter(torch.tensor(weights[0], dtype=torch.float32).T.to(DEVICE))
      self.fc2.weight = torch.nn.parameter.Parameter(torch.tensor(weights[1], dtype=torch.float32).T.to(DEVICE))

  def forward(self, x):
    x = x.to(DEVICE)
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return x


#
# Use the Model class to load your own models into DGA. the run() function
# will be called to test your model. The run() function must return a float
# value representing fitness.
# # # # # # # # # # # # # # # # # # # # # #
class Complex_GA_Model(Model):  # <--- Remember to inherit Model class

  # Load data before running. All file access in load_data is read/write safe, ie. no other subprocess will
  # access the same file at the same time. This is to prevent any OS read/write errors.
  def load_data(self, **kwargs):
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=ToTensor(),
                                         download=True)
    self.testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                             shuffle=False, num_workers=2)

  # Run your model here. The run() function must return a float value representing fitness.
  # Here, we create an ANN which will try to classify handwritten digits from the MNIST dataset.
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


if __name__ == '__main__':
  gene_shapes = {'l1': (28 * 28, 120), 'l2': (120, 10)}
  GA = Genetic_Algorithm(gene_shape=gene_shapes,
                         mutation_rate=0.5,
                         num_genes=10,
                         iterations=100)
  Server(run_name="my_run",
         algorithm=GA,
         model=Complex_GA_Model(),
         num_parallel_processes=2, )
