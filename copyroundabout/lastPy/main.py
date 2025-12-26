from sumolib import checkBinary
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
import xml.etree.ElementTree as ET
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from sumo_env import SumoEnv  
import torch as tt
import CustomNetwork as network
import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CustomWrapper(gym.Wrapper):
    def step(self, action):
        # 从环境 'step' 方法接收五个值
        # result = self.env.step(action)
        # print("CustomWrapper step result:", result)  # 调试信息
        obs, reward, done, truncated, info = self.env.step(action)
        info['turncated'] = truncated  # 确保 info 中包含 truncated 信息
        done = done or truncated  # 如果 done 或者 truncated 任何一个为 True，则 done 为 True
        return obs, reward, done, truncated, info  # 返回五个值

    # 确保重置方法也接受关键字参数
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def main():
    env = SumoEnv()
    env = CustomWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    policy_kwargs = dict(
        features_extractor_class=network.CustomNetwork,
        features_extractor_kwargs=dict(output_dim=2)
    )

    model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise, learning_rate=1e-4, gamma=0.99, batch_size=32)

    num_episodes = 10
    episode_rewards = []
    for episode in range(num_episodes):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            # 环境期望返回五个值
            step_results = env.step(action)
            # print("Main step result:", step_results)  # 调试信息
            obs, rewards, done_array, info_array = step_results
            # 处理向量化环境
            done = done_array[0]
            info = info_array[0]
            truncated = info.get('turncated', False)
            total_reward += rewards[0]
            if 'turncated' in info and info['turncated']:
                print("Episode was truncated!!!!!!!!")
                break
        episode_rewards.append(total_reward)
        print("episode is ======================>>", episode)

    env.close()
    print(total_reward)

        # 绘制奖励函数变化曲线图
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Over Episodes')
    plt.show()

if __name__ == "__main__":
    main()
    print('----------------ALL ---------END-----------------------')