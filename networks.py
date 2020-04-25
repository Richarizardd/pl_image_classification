import torch
from torch import nn

from argparse import Namespace

from torch import nn
import torchvision.models as models

### Custom Networks
class View(nn.Module):
    # https://github.com/pytorch/vision/issues/720
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)

class MNIST_ToyNet(nn.Module):
    def __init__(self, hidden_dim=28)
        super(MNIST_ToyNet, self).__init__()

        self.net = nn.Sequential(
            View((-1, 784)),
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class CIFAR_ToyNet(nn.Module):
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(CIFAR_ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x