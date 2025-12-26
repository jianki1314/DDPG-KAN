import os
import torch as T
from new_sumo_env import SumoEnv
import matplotlib.pyplot as plt
from DDPG import DDPG
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 定义超参数
ALPHA = 0.0001  # Actor 学习率  
BETA = 0.00015  # Critic 学习率
GAMMA = 0.99    # 折扣因子
TAU = 0.005     # 软更新系数
ACTION_NOISE = 0.1  # 动作噪声
BATCH_SIZE = 64  # 采样 batch 大小
MAX_SIZE = 100000  # 经验回放缓冲区大小
CKPT_DIR = 'models/'  # 模型保存路径

# 定义平滑奖励函数
def compute_moving_average(rewards, N=10):
    moving_average = []
    for i in range(len(rewards)):
        if i + 1 < N:
            moving_average.append(sum(rewards[:i+1]) / (i+1))
        else:
            moving_average.append(sum(rewards[i+1-N:i+1]) / N)
    return moving_average

def train_main(MAX_EPISODES):
    log_dir = "../picture"
    os.makedirs(log_dir, exist_ok=True)

    env = SumoEnv(render_mode="rgb_array", max_episodes=MAX_EPISODES, flash_episode=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 创建 DDPG 智能体
    agent = DDPG(alpha=ALPHA, beta=BETA, state_dim=state_dim, action_dim=action_dim, ckpt_dir=CKPT_DIR,
                 observation_space=env.observation_space, action_space=env.action_space)

    episode_rewards = []
    # 训练循环
    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        if np.isnan(state).any():
            raise ValueError("NaN detected in environment state directly from reset")
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if np.isnan(action).any():
                raise ValueError(f"Generated invalid action: {action}")
            done = terminated or truncated  # 判断是否结束
            agent.remember(state, action, reward, next_state, done)  # 存储经验
            agent.learn()  # DDPG更新网络，包含KAN训练
            state = next_state
            total_reward += reward

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        episode_rewards.append(total_reward)
    
    env.close()
    agent.save_models(666)

    # 训练结束后进行平滑处理和绘图
    smooth_reward_data = compute_moving_average(episode_rewards, N=10)
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.6, label='Total Reward')
    plt.plot(smooth_reward_data, 'r', linewidth=2, label='Smoothed Reward')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode Over Training')
    plt.grid(True)
    plt.xlim(left=0)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "reward_kan_env2.png"))

def test(num_episodes):
    print("Starting testing...")
    env = SumoEnv(render_mode="rgb_array", max_episodes=num_episodes, flash_episode=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(alpha=ALPHA, beta=BETA, state_dim=state_dim, action_dim=action_dim, ckpt_dir=CKPT_DIR,
                 observation_space=env.observation_space, action_space=env.action_space)
    agent.load_models(666)
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, train=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
        print(f"结束本回合，回合数为: {episode}")
    env.close()

if __name__ == '__main__':
    os.makedirs(CKPT_DIR + 'Actor', exist_ok=True)
    os.makedirs(CKPT_DIR + 'Target_actor', exist_ok=True)
    os.makedirs(CKPT_DIR + 'Critic', exist_ok=True)
    os.makedirs(CKPT_DIR + 'Target_critic', exist_ok=True)
    # train_main(301)
    test(num_episodes=200)
