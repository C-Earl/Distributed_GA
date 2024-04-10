from DGA.Gene import Parameters
from DGA.Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# SOURCE: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class MNIST_Net(nn.Module):
  def __init__(self):
    super(MNIST_Net, self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
    self.conv1 = nn.Conv2d(1, 1, 5, stride=2)
    self.fc1 = nn.Linear(36, 10)  # 5*5 from image dimension
    # self.fc2 = nn.Linear(100, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
    x = self.fc1(x)
    # x = self.fc2(x)
    return x


class MNIST_Model(Model):
  def __init__(self, **kwargs):
    self.test_data = None
    super().__init__()

  def load_data(self, **kwargs):
    # Download and load the test data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)
    self.test_data = testloader

  def run(self, params: Parameters, **kwargs):
    # Create MNIST_Net model & load params
    MNIST_model = MNIST_Net()
    params_ = {k: torch.tensor(v) for k, v in params.items()}
    MNIST_model.load_state_dict(params_)

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
      for data in self.test_data:
        images, labels = data
        outputs = MNIST_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total

# Run/test MNIST_Net
if __name__ == '__main__':
  from torchvision import datasets, transforms
  from torch.utils.data import DataLoader
  import torch.optim as optim

  # Define a transform to normalize the data
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])

  # Download and load the training data
  trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
  trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

  # Download and load the test data
  testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
  testloader = DataLoader(testset, batch_size=64, shuffle=True)

  # Instantiate the model, loss function and optimizer
  neural_net = MNIST_Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)

  # Train the model
  for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = neural_net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:  # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

  print('Finished Training')

  # Test the model
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = neural_net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(correct)
  print('Accuracy of the network on the test images: %d %%' % (
          100 * correct / total))

  # Save model
  torch.save(neural_net.state_dict(), 'MNIST_model.pth')