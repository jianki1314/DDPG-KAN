import gymnasium as gym
from gymnasium import spaces
from kan import KAN
import torch 
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class CustomKANFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int ):
        super(CustomKANFeaturesExtractor, self).__init__(observation_space, features_dim)

        # 以observation_space.shape[0]作为输入特征数来定义KAN
        self.kan = KAN([observation_space.shape[0], features_dim])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.kan(observations)

# 定义一个新的特征提取器，可以根据你的观察空间进行修改
class IdentityFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space):
        super(IdentityFeaturesExtractor, self).__init__(observation_space, features_dim=1)
        # self.linear = nn.Linear(observation_space.shape[0], observation_space.shape[0])
        self.kan = KAN(layers_hidden=[observation_space.shape[0], 15, 1])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.kan(observations)