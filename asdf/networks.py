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

### MNIST Model
class AttentionMIL_MNIST(nn.Module):
    def __init__(self, L=500, D=128, p=0.0, num_classes=1):
        super(AttentionMIL_MNIST, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            View((-1, 50 * 4 * 4))
        )

        self.fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(p),
            nn.Linear(D, num_classes),
        )

        self.classifier = nn.Sequential(
            nn.Linear(L*num_classes, num_classes),
        )

    
    def forward(self, x):
        x = x.squeeze(0)             # N X 1 X 28 X 28

        H = self.conv(x)
        H = self.fc(H)               # N x L
        
        A = self.attention(H)        # N x K
        A = torch.transpose(A, 1, 0) # K x N
        A = torch.softmax(A, dim=1)  # K X N
        
        M = torch.mm(A, H)           # K x L

        Y = self.classifier(M)
        return Y, 

###
class AttentionMIL_Embed(nn.Module):
    def __init__(self, H=1024, L=512, D=384, p=0.0, num_classes=1):
        super(AttentionMIL_MNIST, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(H, L),
            nn.ReLU(),
            nn.Dropout(p),
        )

        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(p),
            nn.Linear(D, num_classes),
        )

        self.classifier = nn.Sequential(
            nn.Linear(L*num_classes, num_classes),
        )

    
    def forward(self, x):
        H = self.fc(x)               # N x L
        
        A = self.attention(H)        # N x K
        A = torch.transpose(A, 1, 0) # K x N
        A = torch.softmax(A, dim=1)  # K X N
        
        M = torch.mm(A, H)           # K x L

        Y = self.classifier(M)
        return Y, A