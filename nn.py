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
        x = self.fc1(x)
        x = F.elu(x)
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
    
class Generator(torch.nn.Module):
    def __init__(self, n_chan=1, z_dim=100, n_gf=64):
        super(Generator, self).__init__()
        self.x_in = nn.Sequential(
            nn.ConvTranspose2d(z_dim, n_gf * 8, 4, 1, 0, bias=False), # Input: z_dim > Conv
            nn.BatchNorm2d(n_gf * 8),
            nn.ReLU(True)
        )
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(n_gf * 8, n_gf * 4, 4, 2, 1, bias=False), # (n_gf*8) x 4 x 4
            nn.BatchNorm2d(n_gf * 4),
            nn.ReLU(True)
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(n_gf * 4, n_gf * 2, 4, 2, 1, bias=False), # (n_gf*4) x 8 x 8
            nn.BatchNorm2d(n_gf * 2),
            nn.ReLU(True)
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(n_gf * 2, n_gf, 4, 2, 1, bias=False), # (n_gf*2) x 16 x 16
            nn.BatchNorm2d(n_gf),
            nn.ReLU(True)
        )
        self.x_out = nn.Sequential(
            nn.ConvTranspose2d(n_gf, n_chan, 1, 1, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.x_in(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.x_out(x)
        return x
    
class Discriminator(torch.nn.Module):
    def __init__(self, n_chan=1, n_df=64):
        super(Discriminator, self).__init__()
        self.x_in = nn.Sequential(
            nn.Conv2d(n_chan, n_df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l1 = nn.Sequential(
            nn.Conv2d(n_df, n_df * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_df * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(n_df * 2, n_df * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_df * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.x_out = nn.Sequential(
            nn.Conv2d(n_df * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.x_in(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.x_out(x)
        return x.view(-1, 1).squeeze(1)