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
    
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=256):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear (hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True
        
    def forward(self, x):
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))
        mean = self.mean(x)
        logvar = self.var(x)                     
        return mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, output_dim=784, hidden_dim=512, latent_dim=256):
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(latent_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.LeakyReLU(self.linear2(x))
        x = self.LeakyReLU(self.linear1(x))
        x_hat = torch.sigmoid(self.output(x))
        return x_hat

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=256):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def decode(self, x):
        return self.decoder(x)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)    
        z = mean + var*epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar