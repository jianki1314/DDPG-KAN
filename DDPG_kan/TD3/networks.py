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
# print(device)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        alpha: float,
        net_arch: List[int] = [1 , 1],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super(ActorNetwork, self).__init__()

        self.activation_fn = activation_fn
        self.action_space = action_space

        observation_dim  = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.net_arch = [observation_dim + 1, observation_dim + 1]
        # actor_net = create_mlp(observation_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # # Deterministic action
        # self.mu = nn.Sequential(*actor_net)
        # 高效kan
        # self.mu = KAN(
        #     layers_hidden=[observation_dim] + net_arch + [action_dim],
        #     )
        # 原始kan
        self.mu = KAN(width=[observation_dim] + net_arch + [action_dim],grid=5,k=3,seed=1).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha) 
        #初始化权重
        self.apply(weight_init)
        self.to(device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = dict(
            net_arch=self.net_arch,
            features_dim=self.features_dim,
            activation_fn=self.activation_fn,
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        raw_action = self.mu(obs)
        min_val = th.FloatTensor(self.action_space.low).to(device)
        max_val = th.FloatTensor(self.action_space.high).to(device)
        scaled_action = th.tanh(raw_action) * (max_val - min_val) / 2.0 + (max_val + min_val) / 2.0
        return scaled_action

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)
    
    def save_checkpoint(self, checkpoint_file):
        th.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        beta : float,
        net_arch: List[int] = [256, 256],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super(CriticNetwork, self).__init__()
        observation_dim  = observation_space.shape[0]
        action_dim = action_space.shape[0]
        
        # 创建单个 Critic 网络
        self.q_network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn(),
            nn.Linear(net_arch[1], 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # 初始化权重
        self.apply(weight_init)
        self.to(device)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        qvalue_input = th.cat([obs, actions], dim=1).to(obs.device)
        return self.q_network(qvalue_input)  # 返回单个值

    def save_checkpoint(self, checkpoint_file):
        th.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file))


class CriticNetwork_KAN(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        beta : float,
        net_arch: List[int] = [1 , 1],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super(CriticNetwork_KAN, self).__init__()
        observation_dim  = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.net_arch = [observation_dim + action_dim + 1, observation_dim + action_dim + 1]

        # # 创建单个 Critic 网络
        # self.q_network = create_mlp(observation_dim + action_dim, 1, net_arch, activation_fn)
        # self.q_network = nn.Sequential(*self.q_network)
        # 使用 高效KAN 网络
        # self.q_network = KAN(
        #     layers_hidden=[observation_dim + action_dim] + net_arch + [1],
        # )
        # 使用 原始KAN 网络
        self.q_network = KAN(
            width=[observation_dim + action_dim] + net_arch + [1],grid=5,k=3,seed=0
        ).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=beta) 
        # 初始化权重
        self.apply(weight_init)
        self.to(device)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        qvalue_input = th.cat([obs, actions], dim=1)
        return self.q_network(qvalue_input)  # 返回单个值

    def save_checkpoint(self, checkpoint_file):
        th.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file))
