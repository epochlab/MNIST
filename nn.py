#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNN(torch.nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.sm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        x = self.sm(x)
        return x
    
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ELU(),
            nn.MaxPool2d(2)
            )
        self.fc1 = nn.Linear(32*7*7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)