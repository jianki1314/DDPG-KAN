from sumolib import checkBinary
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

# import gym
# from gym.envs.registration import register
# register(
#     id='sumo-v0',
#     entry_point='sumo_env:SumoEnv',
# )

##自定义封装器
class CustomWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = info.get("turncated", False)
        done = done or truncated
        return obs, reward, done, truncated, info

    def reset(self,**kwargs):
        return self.env.reset(**kwargs)

def main_test():
    env = SumoEnv()
    env.reset()
    steps = 200

    for _ in range (steps):
            state = env.get_state()
            traci.simulationStep()
    env.close()      

def main():
    # env = gym.make("sumo-v0",render_mode="human")
    env = SumoEnv()
    env = CustomWrapper(env)
    # env = Monitor(env)
    env = DummyVecEnv([lambda:env])
    env = VecMonitor(env)
    # 实例化噪声对象，给策略添加一定的探索噪声
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    ##构建神经网络智能体##
    policy_kwargs = dict(
        features_extractor_class=network.CustomNetwork,
        features_extractor_kwargs=dict(output_dim=2)
    )
    # 实例化学习算法
    model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise, learning_rate=1e-4,gamma=0.99, batch_size=32 )
    # 设置训练的回合数
    num_episodes = 10
    episode_rewards = []  # 用于存储每个episode的总奖励
    for episode in range(num_episodes):
        done = False
        obs= env.reset()
        total_reward = 0
        while not done:##当车辆到达目的地则进行下一次仿真
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            total_reward += rewards
            # truncated = info.get("truncated", False)
            if done :
                info_0 = info[0]  # 获取第一个环境的info
                if 'truncated' in info_0 and info_0['truncated']:
                    print("Episode was truncated!!!!!!!!")
                break
        episode_rewards.append(total_reward)
        print("epiosde is ======================>>",episode)
    env.close()  
    print(total_reward)

if __name__ == "__main__":

    main()
    
    print('----------------ALL ---------END-----------------------')
