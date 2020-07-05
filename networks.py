from argparse import Namespace

import torch
from torch import nn
import torch.nn.functional as F

def define_net(opt: Namespace):
    if opt.net_arch == 'mnist_fc':
        return MNIST_FC()
    elif opt.net_arch == 'mnist_cnn':
        return MNIST_CNN()
    elif opt.net_arch == 'cifar_toy':
        return CIFAR_ToyNet()
    else:
        raise ValueError('Undefined model type')

### Models
class View(nn.Module):
    # https://github.com/pytorch/vision/issues/720
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)


class MNIST_FC(nn.Module):
    def __init__(self, hidden_dim=28):
        super(MNIST_FC, self).__init__()

        self.net = nn.Sequential(
            View((-1, 784)),
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output