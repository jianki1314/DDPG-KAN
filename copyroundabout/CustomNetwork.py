import gymnasium as gym
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


##创建自定义网络##
class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, output_dim: int):
        super(CustomNetwork, self).__init__(observation_space, output_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

