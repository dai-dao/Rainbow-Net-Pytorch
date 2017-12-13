import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable

import math


class DistMLP(nn.Module):
    def __init__(self, hidden, nb_atoms, ob_shape, num_action):
        super(DistMLP, self).__init__()
        self.nb_atoms = nb_atoms
        self.num_action = num_action
        self.l1 = nn.Linear(ob_shape, hidden)
        self.phi = nn.Linear(hidden, num_action * nb_atoms)

    def forward(self, x):
        x = F.relu(self.l1(x))
        out = self.phi(x).view(-1, self.num_action, self.nb_atoms)
        out = F.softmax(out, dim=-1)
        return out


class DistDuelingMLP(nn.Module):
    def __init__(self, hidden, nb_atoms, ob_shape, num_action):
        super(DistDuelingMLP, self).__init__()
        self.nb_atoms = nb_atoms
        self.num_action = num_action
        self.l1 = nn.Linear(ob_shape, hidden)
        self.value = nn.Linear(hidden, nb_atoms)
        self.advantage = nn.Linear(hidden, num_action * nb_atoms)

    def forward(self, x):
        x = F.relu(self.l1(x))

        value = self.value(x).view(-1, self.nb_atoms).unsqueeze(1).expand(-1, self.num_action, self.nb_atoms)
        advantage = self.advantage(x).view(-1, self.num_action, self.nb_atoms)
        phi = value + (advantage - advantage.mean(1, keepdim=True))
        out = F.softmax(phi, dim=-1)
        return out


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)

        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))

        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    
    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant(self.sigma_weight, self.sigma_init)
            init.constant(self.sigma_bias, self.sigma_init)

    
    def forward(self, x):
        return F.linear(x, self.weight + self.sigma_weight * Variable(self.epsilon_weight), 
                           self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    
    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    
    def reset_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


    
class NoisyDistDuelingMLP(nn.Module):
    def __init__(self, hidden, nb_atoms, ob_shape, num_action):
        super(NoisyDistDuelingMLP, self).__init__()
        self.nb_atoms = nb_atoms
        self.num_action = num_action
        self.l1 = nn.Linear(ob_shape, hidden)
        self.value = NoisyLinear(hidden, nb_atoms)
        self.advantage = NoisyLinear(hidden, num_action * nb_atoms)


    def forward(self, x):
        x = F.relu(self.l1(x))
        value = self.value(x).view(-1, self.nb_atoms).unsqueeze(1).expand(-1, self.num_action, self.nb_atoms)
        advantage = self.advantage(x).view(-1, self.num_action, self.nb_atoms)
        phi = value + (advantage - advantage.mean(1, keepdim=True))
        out = F.softmax(phi, dim=-1)
        return out


    def sample_noise(self):
        self.value.sample_noise()
        self.advantage.sample_noise()

    
    def reset_noise(self):
        self.value.reset_noise()
        self.advantage.reset_noise()