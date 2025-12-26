from sumolib import checkBinary
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
import xml.etree.ElementTree as ET
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback,StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.utils import polyak_update
from new_sumo_env import SumoEnv  
import NetworkKAN as kan
import torch
import torch.nn.functional as F
import numpy as np
###此函数为  训练 +  测试  主函数###

def compute_moving_average(rewards, N=10):
    moving_average = []
    for i in range(len(rewards)):
        if i+1 < N:
            moving_average.append(sum(rewards[:i+1]) / (i+1))
        else:
            moving_average.append(sum(rewards[i+1-N:i+1]) / N)
    return moving_average


def main():
    net_arch = [15, 15]  
    kan_params = dict(grid_size=5, spline_order=3, scale_noise=0.1) 

    log_dir = "/tmp/"
    model_path = "/home/jian/sumo/copyroundabout/model"
    env = SumoEnv(render_mode='rgb_array', max_episodes=100, flash_episode=False)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # 使用VecMonitor监控环境
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = DDPG("MlpPolicy", 
                env, verbose=1, 
                policy_kwargs=dict(
                    features_extractor_class=kan.IdentityFeaturesExtractor, 
                    net_arch=net_arch  # 确保net_arch仅包含整数列表
                ), 
                action_noise=action_noise,
                learning_rate=1e-4,
                gamma=0.99, 
                batch_size=32,
                buffer_size=5000)
    
    print("开始模型学习")
    model.learn(total_timesteps=1000)
    env.close()
    
    # 学习完成后保存模型
    model.save(os.path.join(model_path, "DDPG_kan_model"))

    # 加载监控数据并提取奖励信息
    monitor_data = load_results(log_dir)
    reward_data = monitor_data['r']
    smooth_reward_data = compute_moving_average(reward_data,N=10)
    plt.figure(figsize=(12, 6))
    # 原始奖励曲线
    plt.plot(reward_data, alpha=0.6, label='Total Reward')  # alpha值用于调整线的透明度
    
    # 平滑奖励曲线
    plt.plot(smooth_reward_data, 'r', linewidth=2, label='Smoothed Reward')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode Over Training')
    plt.grid(True)
    plt.xlim(left=0)
    
    plt.savefig(os.path.join(log_dir,"/home/jian/sumo/copyroundabout/picture/reward_ddpg_kan.png"))  # 如果想保存图片到文件
    # plt.show()
    print("$$$$$$$$$  模型学习完毕   $$$$$$$$$$$$")

    #开启实际仿真
    print("###############      训练结束  #####################################################################")
    log_file = os.path.join("/tmp", "monitor.csv")
    # 检查文件是否存在，如果是，则删除
    if os.path.isfile(log_file):
        os.remove(log_file)

    # 现在创建并包装您的环境，会创建一个新的日志文件
    env = SumoEnv(render_mode='rgb_array', max_episodes=100, flash_episode=True)
    # 对环境进行包装
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=log_file)

    # for env_i in env.envs:
    #     env_i.flash_episode = True
    model = DDPG.load(os.path.join(model_path,"DDPG_kan_model"))
    total_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done_array, info_array = env.step(action)
        info = info_array[0]
        truncated = info.get('message', False) ##截断条件
        done = truncated
        total_reward += rewards[0]             
    env.close()  

def test_model():
    log_file = os.path.join("/tmp", "monitor.csv")
    # 检查文件是否存在，如果是，则删除
    if os.path.isfile(log_file):
        os.remove(log_file)
    model_path = "/home/jian/sumo/copyroundabout/model"
        # 现在创建并包装您的环境，会创建一个新的日志文件
    env = SumoEnv(render_mode='rgb_array', max_episodes=100, flash_episode=True)
    # 对环境进行包装
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=log_file)

    # for env_i in env.envs:
    #     env_i.flash_episode = True
    model = DDPG.load(os.path.join(model_path,"kan_model"))
    total_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done_array, info_array = env.step(action)
        info = info_array[0]
        truncated = info.get('message', False) ##截断条件
        done = truncated
        total_reward += rewards[0]             
    env.close() 

class CustomDDPG(DDPG):
    def train(self, gradient_steps: int, batch_size: int = 64):
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # 1. 预测下一个状态的动作
                target_actions = self.actor_target(replay_data.next_observations)
                # 2. 计算目标网络对下一个状态的评估
                target_q_values = self.critic_target(replay_data.next_observations, target_actions)
                # 3. 计算期望的Q值
                target_q_values = replay_data.rewards + (1 - replay_data.dones.float()) * self.gamma * target_q_values

            # 4. 计算当前Q值
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            # 5. 计算损失
            critic_loss = F.mse_loss(current_q_values, target_q_values)

            # 6. 优化评价者
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic.optimizer.step()

            # 7. 延迟策略更新
            if gradient_step % self.policy_delay == 0:
                for param in self.critic.parameters():
                    param.requires_grad = False

                # 计算演员损失
                actor_loss = -self.critic(replay_data.observations, self.actor(replay_data.observations)).mean()

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor.optimizer.step()

                for param in self.critic.parameters():
                    param.requires_grad = True

                # 更新目标网络
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

def train():
    net_arch = [13, 13]  
    kan_params = dict(grid_size=5, spline_order=3, scale_noise=0.1) 

    log_dir = "/tmp/"
    model_path = "/home/jian/sumo/copyroundabout/model"
    env = SumoEnv(render_mode='rgb_array', max_episodes=100, flash_episode=False)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # 使用VecMonitor监控环境
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = CustomDDPG("MlpPolicy", 
                env, verbose=1, 
                policy_kwargs=dict(
                    features_extractor_class=kan.IdentityFeaturesExtractor, 
                    net_arch=net_arch  # 确保net_arch仅包含整数列表
                ), 
                action_noise=action_noise,
                learning_rate=1e-4,
                gamma=0.99, 
                batch_size=32,
                buffer_size=1000)
    
    print("开始模型学习")
    model.learn(total_timesteps=1000)
    env.close()
    
    # 学习完成后保存模型
    model.save(os.path.join(model_path, "DDPG_kan_model"))

    # 加载监控数据并提取奖励信息
    monitor_data = load_results(log_dir)
    reward_data = monitor_data['r']
    smooth_reward_data = compute_moving_average(reward_data,N=10)
    plt.figure(figsize=(12, 6))
    # 原始奖励曲线
    plt.plot(reward_data, alpha=0.6, label='Total Reward')  # alpha值用于调整线的透明度
    
    # 平滑奖励曲线
    plt.plot(smooth_reward_data, 'r', linewidth=2, label='Smoothed Reward')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode Over Training')
    plt.grid(True)
    plt.xlim(left=0)
    
    plt.savefig(os.path.join(log_dir,"/home/jian/sumo/copyroundabout/picture/reward_ddpg_kan.png"))  # 如果想保存图片到文件
    # plt.show()
    print("$$$$$$$$$  模型学习完毕   $$$$$$$$$$$$")

if __name__ == "__main__":
    main()
    # test_model()
    # train()
    print('----------------ALL ---------END-----------------------')
