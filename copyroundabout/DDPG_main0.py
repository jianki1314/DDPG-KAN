from sumolib import checkBinary
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
import xml.etree.ElementTree as ET
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from new_sumo_env import SumoEnv  
import torch as tt
import CustomNetwork as network
import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces
cfg_path = "networks/sumoconfig.sumo.cfg"
###旧版本 DDPG-main#####
# class CustomWrapper(gym.Wrapper):
#     def step(self, action):
#         # 从环境 'step' 方法接收五个值
#         # result = self.env.step(action)
#         # print("CustomWrapper step result:", result)  # 调试信息
#         obs, reward, done, truncated, info = self.env.step(action)
#         #info['truncated'] = truncated  # 确保 info 中包含 truncated 信息
#         # done = done or truncated  # 如果 done 或者 truncated 任何一个为 True，则 done 为 True
#         return obs, reward, done, truncated, info  # 返回五个值   
    
#     # 确保重置方法也接受关键字参数
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

##对输出文件更名##
def modify_output_prefix(cfg_file, new_prefix):
    tree = ET.parse(cfg_file)
    root = tree.getroot()
    output_prefix = root.find('.//output-prefix')
    if output_prefix is not None:
        output_prefix.set('value', new_prefix)
    tree.write(cfg_file)

def main():
    log_dir = "/tmp/"
    env = SumoEnv(render_mode='rgb_array')
    # 若有必要，env 可以进一步经过其他自定义 Wrapper 封装
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # 使用VecMonitor监控环境
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise,
                 learning_rate=1e-4, gamma=0.99, batch_size=32, buffer_size=1000)
    
    print("开始模型学习")
    model.learn(total_timesteps=5000)
    # env.close()
    
    # 学习完成后保存模型（如果需要）
    # model.save(os.path.join(log_dir, "ddpg_model"))

    # 加载监控数据并提取奖励信息
    monitor_data = load_results(log_dir)
    reward_data = monitor_data['r']

    plt.figure(figsize=(12, 6))
    plt.plot(reward_data)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode Over Training')
    plt.grid(True)
    plt.xlim(left=0)
    
    plt.savefig(os.path.join(log_dir,"/home/jian/sumo/copyroundabout/picture/reward_noise-0.2.png"))  # 如果想保存图片到文件
    # plt.show()
    print("$$$$$$$$$  模型学习完毕   $$$$$$$$$$$$")

    #开启实际仿真
    # num_episodes = 300
    # episode_rewards = []
    # for episode in range(num_episodes):
    #     modify_output_prefix(cfg_path,f'{episode}-')
    #     done = False
    #     obs = env.reset()
    #     total_reward = 0
    #     while not done:
    #         action, _states = model.predict(obs)
    #         step_results = env.step(action)
    #         obs, rewards, done_array, info_array = step_results
    #         #done = done_array[0]
    #         info = info_array[0]
    #         truncated = info.get('truncated', False)
    #         done = truncated
    #         total_reward += rewards[0]             
    #         if info.get('truncated', False):
    #             break  # 如果检测到仿真被截断，则退出while循环以结束本回合
    #         # traci.simulationStep()
    #     env.close()  # 这将自动保存已经指定好的路由和旅行信息文件
    #     episode_rewards.append(total_reward)
    #     print("episode is ======================>>", episode)

    # print(total_reward)
    # plt.plot(episode_rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Total Rewards Over Episodes')
    # plt.grid(True)
    # plt.xlim(left=0)
    # #plt.show()
    # save_path = '/home/jian/sumo/copyroundabout/picture/reward_300.png'  # 修改为您要保存的路径
    # plt.savefig(save_path)

def run_model(action_noise_level):
    log_dir = "/tmp/"
    env = SumoEnv(render_mode='rgb_array')
    env = DummyVecEnv([lambda: env])
    monitored_env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    
    n_actions = monitored_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=action_noise_level * np.ones(n_actions))

    model = DDPG("MlpPolicy", monitored_env, verbose=1, action_noise=action_noise,
                 learning_rate=1e-4, gamma=0.99, batch_size=32, buffer_size=1000)
    
    print(f"开始模型学习，动作噪声水平为： {action_noise_level}")
    model.learn(total_timesteps=5000)
    env.close()
    
    monitor_data = load_results(log_dir)
    
    # 返回获取到的回合累积奖励数据
    return monitor_data['r']

def main_find_best_noise():
    noise_levels = [0.0, 0.1, 0.2, 0.4, 0.5 ]  # 假设有三个不同的动作噪声水平
    plt.figure(figsize=(12, 6))
    reward_stats ={ }
    
    for noise in noise_levels:
        reward_data = run_model(noise)
        plt.plot(reward_data, label=f"Noise {noise}")
        # 计算并存储每个噪声水平下奖励值的统计信息
        mean_reward = np.mean(reward_data)  # 平均值
        var_reward = np.var(reward_data)   # 方差
        
        reward_stats[f'Noise {noise}'] = {'mean': mean_reward, 'variance': var_reward}

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode Over Training With Different Noise Levels')
    
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.xlim(left=0)

    plt.savefig("/home/jian/sumo/copyroundabout/picture/reward_comparison.png")  
    plt.show()
    # 打印不同噪声水平下奖励值的统计信息
    print("Reward Statistics (Mean and Variance) under Different Noise Levels:")
    
    for level_stat in reward_stats.keys():
        print(f"{level_stat} - Mean: {reward_stats[level_stat]['mean']}, Variance: {reward_stats[level_stat]['variance']}")


if __name__ == "__main__":
    main()
    print('----------------ALL ---------END-----------------------')
