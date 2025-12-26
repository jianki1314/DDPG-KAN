import torch as th
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torch.distributions import Normal
from kan import KAN
import numpy as np
from gymnasium import spaces
LOG_STD_MAX = 2
LOG_STD_MIN = -20
th.use_deterministic_algorithms(False)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
print(device)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch, activation_fn=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(net_arch)):
            layers.append(nn.Linear(input_dim, net_arch[i]))
            layers.append(activation_fn())
            input_dim = net_arch[i]
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, alpha, net_arch=[256, 256], activation_fn=nn.ReLU):
        super(ActorNetwork, self).__init__()
        observation_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.action_space = action_space
        self.actor = MLP(observation_dim, action_dim, net_arch, activation_fn)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.apply(weight_init)
        self.to(device)

    def forward(self, obs):
        raw_action = self.actor(obs)
        min_val = th.FloatTensor(self.action_space.low).to(device)
        max_val = th.FloatTensor(self.action_space.high).to(device)
        scaled_action = th.tanh(raw_action) * (max_val - min_val) / 2.0 + (max_val + min_val) / 2.0
        return scaled_action
    
    def save_checkpoint(self, checkpoint_file):
        th.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, observation_space, action_space, beta, net_arch=[256, 256], activation_fn=nn.ReLU):
        super(CriticNetwork, self).__init__()
        observation_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        input_dim = observation_dim + action_dim
        self.critic = MLP(input_dim,1, net_arch, activation_fn)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.apply(weight_init)
        self.to(device)

    def forward(self, obs, actions):
        qvalue_input = th.cat([obs, actions], dim=1)
        return self.critic(qvalue_input)
    
    def save_checkpoint(self, checkpoint_file):
        th.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file))
