from sumolib import checkBinary
import os  
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3 import TD3,SAC
from stable_baselines3 import PPO
import xml.etree.ElementTree as ET
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback,StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import load_results
# from new_sumo_env2 import SumoEnv 
from new_sumo_env import SumoEnv 
import tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces
###此函数为  训练 +  测试  主函数### 最新的

def compute_moving_average(rewards, N=10):
    moving_average = []
    for i in range(len(rewards)):
        if i+1 < N:
            moving_average.append(sum(rewards[:i+1]) / (i+1))
        else:
            moving_average.append(sum(rewards[i+1-N:i+1]) / N)
    return moving_average


def main_DDPG():
    log_dir = tempfile.gettempdir()
    model_path = "/home/jian/sumo/copyroundabout/model"
    sumocfg_config = "/home/jian/sumo/copyroundabout/networks/sumoconfig_DDPG.sumo.cfg"
    env = SumoEnv(render_mode='rgb_array', max_episodes=300, flash_episode=False)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # 使用VecMonitor监控环境
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise,
                 learning_rate=1e-4, gamma=0.99, batch_size=64, buffer_size=100000)
    
    print("开始模型学习")
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=200,verbose=1)
    model.learn(total_timesteps=100000, callback=callback_max_episodes)
    env.close()
    
    # 学习完成后保存模型
    model.save(os.path.join(model_path, "DDPG_model"))

    # 加载监控数据并提取奖励信息
    monitor_data = load_results(log_dir)
    reward_data = monitor_data['r']
    smooth_reward_data = compute_moving_average(reward_data,N=10)
    plt.figure(figsize=(12, 6))
    # 原始奖励曲线
    plt.plot(reward_data, alpha=1, label='Total Reward')  # alpha值用于调整线的透明度
    
    # 平滑奖励曲线
    plt.plot(smooth_reward_data, 'r', linewidth=2, label='Smoothed Reward')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode Over Training')
    plt.grid(True)
    plt.xlim(left=0)
    
    plt.savefig(os.path.join(log_dir,"/home/jian/sumo/copyroundabout/picture/reward_DDPG.png"))  # 如果想保存图片到文件
    # plt.show()
    print("$$$$$$$$$  模型学习完毕   $$$$$$$$$$$$")

    #开启实际仿真
    
    print("###############      仿真开启  #####################################################################")
    log_file = os.path.join(log_dir, "monitor.csv")
    # 检查文件是否存在，如果是，则删除
    if os.path.isfile(log_file):
        os.remove(log_file)

    # 现在创建并包装您的环境，会创建一个新的日志文件
    env = SumoEnv(render_mode='rgb_array', max_episodes=200, flash_episode=True)
    # 对环境进行包装
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=log_file)

    model = DDPG.load(os.path.join(model_path,"DDPG_model"))
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done_array, info_array = env.step(action)
        info = info_array[0]
        done = info.get('message', 'False') == 'True'              
    env.close()  # 这将自动保存已经指定好的路由和旅行信息文件


    


def test():
    model_path = "/home/jian/sumo/copyroundabout/model"
    log_file = os.path.join("/tmp", "monitor.csv")
    # 现在创建并包装您的环境，会创建一个新的日志文件
    env = SumoEnv(render_mode='rgb_array', max_episodes=100, flash_episode=True)
    # 对环境进行包装
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=log_file)

    # for env_i in env.envs:
    #     env_i.flash_episode = True
    # model = PPO.load(os.path.join(model_path,"PPO1_model"))
    # model = DDPG.load(os.path.join(model_path,"DDPG_model"))
    model = TD3.load(os.path.join(model_path,"TD3_model"))
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
    env.close()  # 这将自动保存已经指定好的路由和旅行信息文件


def main_TD3():
    log_dir = tempfile.gettempdir()
    model_path = "/home/jian/sumo/copyroundabout/model"
    env = SumoEnv(render_mode='rgb_array', max_episodes=300, flash_episode=False)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # 使用VecMonitor监控环境
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, verbose=1, action_noise=action_noise,
                learning_rate=1e-4, gamma=0.99, batch_size=64, buffer_size=100000)

    print("开始模型学习")
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=300,verbose=1)
    model.learn(total_timesteps=100000, callback=callback_max_episodes)
    env.close()
    
    # 学习完成后保存模型
    model.save(os.path.join(model_path, "TD3_model"))

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
    
    plt.savefig(os.path.join(log_dir,"/home/jian/sumo/copyroundabout/picture/reward_TD3.png"))  # 如果想保存图片到文件
    # plt.show()
    print("$$$$$$$$$  模型学习完毕   $$$$$$$$$$$$")

    #开启实际仿真
    print("###############      训练结束  #####################################################################")
    log_file = os.path.join("/tmp", "monitor.csv")
    # 检查文件是否存在，如果是，则删除
    if os.path.isfile(log_file):
        os.remove(log_file)

    # 现在创建并包装您的环境，会创建一个新的日志文件
    env = SumoEnv(render_mode='rgb_array', max_episodes=201, flash_episode=True)
    # 对环境进行包装
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=log_file)

    model = TD3.load(os.path.join(model_path,"TD3_model"))
    total_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done_array, info_array = env.step(action)
        info = info_array[0]
        done = info.get('message', 'False') == 'True'              
    env.close()  # 这将自动保存已经指定好的路由和旅行信息文件

def main_SAC():
    log_dir = tempfile.gettempdir()
    model_path = "/home/jian/sumo/copyroundabout/model"
    env = SumoEnv(render_mode='rgb_array', max_episodes=300, flash_episode=False)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # 使用VecMonitor监控环境
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = SAC("MlpPolicy", env, verbose=1, action_noise=action_noise,
                learning_rate=1e-4, gamma=0.99, batch_size=64, buffer_size=100000)

    print("开始模型学习")
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=300,verbose=1)
    model.learn(total_timesteps=100000, callback=callback_max_episodes)
    env.close()
    
    # 学习完成后保存模型
    model.save(os.path.join(model_path, "SAC_model"))

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
    
    plt.savefig(os.path.join(log_dir,"/home/jian/sumo/copyroundabout/picture/reward_SAC.png"))  # 如果想保存图片到文件
    # plt.show()
    print("$$$$$$$$$  模型学习完毕   $$$$$$$$$$$$")

    #开启实际仿真
    print("###############      开始仿真  #####################################################################")
    log_file = os.path.join(log_dir, "monitor.csv")
    # 检查文件是否存在，如果是，则删除
    if os.path.isfile(log_file):
        os.remove(log_file)

    # 现在创建并包装您的环境，会创建一个新的日志文件
    env = SumoEnv(render_mode='rgb_array', max_episodes=201, flash_episode=True)
    # 对环境进行包装
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=log_file)

    model = SAC.load(os.path.join(model_path,"SAC_model"))
    total_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done_array, info_array = env.step(action)
        info = info_array[0]
        done = info.get('message', 'False') == 'True'              
    env.close()  # 这将自动保存已经指定好的路由和旅行信息文件

def main_PPO():
    log_dir = tempfile.gettempdir()
    model_path = "/home/jian/sumo/copyroundabout/model"
    env = SumoEnv(render_mode='rgb_array', max_episodes=300, flash_episode=False)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # 使用VecMonitor监控环境
    
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=1e-4,  # PPO的学习率通常会设置得较低
                gamma=0.99,
                n_steps=1024,  # 每1024步更新一次模型
                batch_size=64,
                n_epochs=10,  # 每次学习循环更新模型的次数
                clip_range=0.2,  # 用于限制策略更新的参数
                device='cpu')
    
    print("开始模型学习")
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=300, verbose=1)
    model.learn(total_timesteps=100000, callback=callback_max_episodes)
    env.close()
    # 学习完成后保存模型
    model.save(os.path.join(model_path, "PPO_model"))

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
    
    plt.savefig(os.path.join(log_dir,"/home/jian/sumo/copyroundabout/picture/reward_PPO.png"))  # 如果想保存图片到文件
    # plt.show()
    print("$$$$$$$$$  模型学习完毕   $$$$$$$$$$$$")

    #开启实际仿真
    print("###############      训练结束  #####################################################################")
    log_file = os.path.join(log_dir, "monitor.csv")
    # 检查文件是否存在，如果是，则删除
    if os.path.isfile(log_file):
        os.remove(log_file)

    # 现在创建并包装您的环境，会创建一个新的日志文件
    env = SumoEnv(render_mode='rgb_array', max_episodes=201, flash_episode=True)
    # 对环境进行包装
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, filename=log_file)

    model = PPO.load(os.path.join(model_path,"PPO_model"))
    total_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done_array, info_array = env.step(action)
        info = info_array[0]
        done = info.get('message', 'False') == 'True'              
    env.close()  # 这将自动保存已经指定好的路由和旅行信息文件           

if __name__ == "__main__":
    main_DDPG()
    # main_PPO()
    # main_SAC()
    # main_TD3()
    # test()
    print('----------------ALL ---------END-----------------------')
