import os
import torch as T
from new_sumo_env import SumoEnv
import matplotlib.pyplot as plt
from TD3 import TD3
from kan import KAN
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 定义超参数
ALPHA = 0.0001 # Actor 学习率
BETA = 0.00015 # Critic 学习率
GAMMA = 0.99  # 折扣因子
TAU = 0.005  # 软更新系数
ACTION_NOISE = 0.1 # 动作噪声
BATCH_SIZE = 64  # 采样 batch 大小
MAX_SIZE = 100000 # 经验回放缓冲区大小
CKPT_DIR = 'models/'  # 模型保存路径

# 定义自定义激活函数（将输出限制在 [-3, 4] 之间）
def custom_activation(x):
    x = T.tanh(x)  # 输出范围 [-1, 1]
    return 3.5 * x + 0.5  # 调整范围到 [-3, 4]

#定义平滑奖励 函数
def compute_moving_average(rewards, N=10):
    moving_average = []
    for i in range(len(rewards)):
        if i+1 < N:
            moving_average.append(sum(rewards[:i+1]) / (i+1))
        else:
            moving_average.append(sum(rewards[i+1-N:i+1]) / N)
    return moving_average

def train_main(MAX_EPISODES):
    log_dir = "../picture" 
    os.makedirs(log_dir, exist_ok=True)  # 如果目录不存在，则创建

    env = SumoEnv(render_mode="rgb_array", max_episodes=MAX_EPISODES, flash_episode=False)
    state_dim = env.observation_space.shape[0]  # 获取状态空间维度
    action_dim = env.action_space.shape[0]  # 获取动作空间维度
    # 创建 DDPG 智能体
    agent = TD3(alpha=ALPHA, beta=BETA, state_dim=state_dim, action_dim=action_dim, observation_space=env.observation_space,action_space=env.action_space,ckpt_dir=CKPT_DIR)
    
    episode_rewards = [] 
    # 训练循环
    for episode in range(MAX_EPISODES):
        state, _ = env.reset()  # 重置环境
        if np.isnan(state).any():
            raise ValueError("NaN detected in environment state directly from reset")
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if np.isnan(action).any():
                raise ValueError(f"Generated invalid action: {action}")
            done = terminated or truncated #判断是否结束
            agent.remember(state, action, reward, next_state, done)  # 存储经验
            agent.learn()  # 更新网络
            state = next_state
            total_reward += reward
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        episode_rewards.append(total_reward)
    env.close()  # 关闭环境
    agent.save_models(2)


    # 训练结束后进行平滑处理和绘图
    smooth_reward_data = compute_moving_average(episode_rewards, N=10) 
    plt.figure(figsize=(12, 6))
    # 原始奖励曲线
    plt.plot(episode_rewards, alpha=0.6, label='Total Reward')  
    # 平滑奖励曲线
    plt.plot(smooth_reward_data, 'r', linewidth=2, label='Smoothed Reward')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode Over Training')
    plt.grid(True)
    plt.xlim(left=0)
    plt.legend() # 显示图例  
    # 保存图片到文件
    plt.savefig(os.path.join(log_dir, "reward_TD3-kan.png"))  
    # plt.show() # 如果需要显示图片



def test(num_episodes):
    print("Starting testing...")
    env = SumoEnv(render_mode="rgb_array", max_episodes=num_episodes, flash_episode=True)
    state_dim = env.observation_space.shape[0]  # 获取状态空间维度
    action_dim = env.action_space.shape[0]  # 获取动作空间维度
    agent = TD3(alpha=ALPHA, beta=BETA, state_dim=state_dim, action_dim=action_dim, observation_space=env.observation_space,action_space=env.action_space,ckpt_dir=CKPT_DIR)
    agent.load_models(2)
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, train=True)  # 取消探索噪声
            # print(f"Action is: {action}, Action shape is: {action.shape}")
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
        print("结束本回合，回合数为============>>",episode)
    env.close()   

if __name__ == '__main__':
    os.makedirs(CKPT_DIR + 'Actor', exist_ok=True)
    os.makedirs(CKPT_DIR + 'Target_actor', exist_ok=True)
    os.makedirs(CKPT_DIR + 'Critic1', exist_ok=True)
    os.makedirs(CKPT_DIR + 'Target_critic1', exist_ok=True)
    os.makedirs(CKPT_DIR + 'Critic2', exist_ok=True)
    os.makedirs(CKPT_DIR + 'Target_critic2', exist_ok=True)
    train_main(301)
    test(num_episodes=200)