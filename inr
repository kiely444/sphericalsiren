import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from spherical_harmonics_ylm import SH
import pyshtools as pysh

class SHneuron(nn.Module):
    def __init__(self, lmax, gofx):
        super(SHneuron, self).__init__()
        self.lmax = lmax
        self.n = int((self.lmax + 1)**2)  # number of nonzero coeffs
        self.linear = nn.Linear(self.n, self.n, dtype=torch.float64)
        with torch.no_grad():
            init.normal_(self.linear.weight, mean=0, std=1)
            index = 0
            for l in range(self.lmax+1):
                if l == 0: power = 1.0
                else: power = l**-2
                for i in range(l*2+1):
                    self.linear.weight[index] = self.linear.weight[index] * power
                    index += 1
        self.gofx = gofx
    
    def forward(self, theta_phi):
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]
        self.shcoeffs = self.linear(torch.ones(self.n, dtype=torch.float64))
        output = torch.zeros(theta_phi.size(0), dtype=torch.float64)  # output tensor
        coeff_index = 0
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                sh_value = SH(m, l, phi, theta)
                output += sh_value * self.shcoeffs[coeff_index]
                coeff_index += 1
        return self.gofx(output)

class SHlayer(nn.Module):
    def __init__(self, lmax, L1neurons, gofx):
        super().__init__()
        self.L1neurons = L1neurons
        self.layer = nn.ModuleList([SHneuron(lmax, gofx) for _ in range(L1neurons)])
    
    def forward(self, theta_phi):
        output = torch.zeros(theta_phi.size(0), self.L1neurons, dtype=torch.float64)
        for i in range(self.L1neurons):
            output[:, i] = self.layer[i](theta_phi)
        return output

class HiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, gofx, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float64)
        with torch.no_grad():
            self.linear.weight.normal_(mean=0, std=0.01)
        self.gofx = gofx

    def forward(self, input):
        return self.gofx(self.linear(input))

class SphericalSiren(nn.Module):
    def __init__(self, lmax, neurons, hidden_layers, useSine=False):
        super().__init__()
        self.useSine = useSine
        
        if useSine:
            self.gofx = torch.sin
        else:
            self.gofx = lambda x: 1.0 - (torch.square(x) / 2)
        
        layers = []
        layers.append(SHlayer(lmax, neurons, self.gofx))
        in_features = neurons
        for i in range(hidden_layers):
            layers.append(HiddenLayer(in_features, neurons, self.gofx))
            in_features = neurons
        final_layer = nn.Linear(neurons, 1, dtype=torch.float64)
        with torch.no_grad():
            final_layer.weight.normal_(mean=0, std=0.01)
            final_layer.bias.normal_(mean=0, std=0.01)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)
        print (f'Number of neurons: {neurons}, number of layers:{hidden_layers}, useSine={useSine}')
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        out = self.network(coords)
        return out, coords
