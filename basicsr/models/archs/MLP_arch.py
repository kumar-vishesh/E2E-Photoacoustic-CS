# Simple SIREN architecture for demultiplexing
import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega_0=30, is_first=False):
        super().__init__()
        self.in_dim = in_dim
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_dim, 1 / self.in_dim)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_dim) / self.omega_0,
                    np.sqrt(6 / self.in_dim) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class MLP(nn.Module):
    def __init__(self, size, hidden_layers, omega_0):
        super().__init__()

        in_dim = size
        hidden_dim = in_dim
        out_dim = size

        layers = [SineLayer(in_dim, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        layers.append(nn.Linear(hidden_dim, out_dim))
        with torch.no_grad():
            layers[-1].weight.uniform_(-1e-4, 1e-4)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)