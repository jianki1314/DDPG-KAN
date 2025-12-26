import torch as T
import torch.nn.functional as F
import numpy as np
from noise import OUActionNoise
from networks import ActorNetwork, CriticNetwork
# from torch.utils.tensorboard import SummaryWriter
from buffer import ReplayBuffer
from kan import KAN

 
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print(device)
 
##检查nan值函数
def check_nan(tensor, name):
    if T.isnan(tensor).any():
        print(f"NaN detected in {name}")

 
class DDPG:
    def __init__(self, alpha, beta, state_dim, action_dim, ckpt_dir, observation_space, action_space,
                 gamma=0.99, tau=0.005, action_noise=0.3, max_size=131072,batch_size=256):
        # self.writer = SummaryWriter()  # 初始化 SummaryWriter
        self.global_step = 0  # 添加此行来定义 global_step
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.action_space = action_space
        self.observation_space = observation_space

        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        # self.ou_noise = OUActionNoise(mu=np.zeros(action_dim), sigma=action_noise)
        self.ou_noise = OUActionNoise(mu=np.zeros(action_dim), sigma=action_noise, decay=0.99)##加入噪声衰减
        self.checkpoint_dir = ckpt_dir
        # self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim)
        # self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim)
        # self.critic = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,fc1_dim=fc1_dim, fc2_dim=fc2_dim, action_space=action_space)
        # self.target_critic = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,fc1_dim=fc1_dim, fc2_dim=fc2_dim, action_space=action_space)
        self.actor = ActorNetwork(observation_space, action_space, alpha).to(device)
        self.target_actor = ActorNetwork(observation_space, action_space, alpha).to(device)
        self.critic = CriticNetwork(observation_space, action_space, beta).to(device)
        self.target_critic = CriticNetwork(observation_space, action_space, beta).to(device)
 
        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,batch_size=batch_size)
 
        self.update_network_parameters(tau=self.tau)


    def update_network_parameters(self, tau=None):
        if tau is None:
            # tau = self.tau
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                        self.target_actor.parameters()):
                target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)
    
            for critic_params, target_critic_params in zip(self.critic.parameters(),
                                                        self.target_critic.parameters()):
                target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)
 
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
 
    def choose_action(self, observation, train=True):
        if np.isnan(observation).any():
            raise ValueError("NaN detected in observation passed to the Actor network")
        
        self.actor.eval()
        
        # 将 observation 转换为单个 NumPy 数组，并确保在 CUDA 设备上
        state = T.tensor(np.array(observation, dtype=np.float32), dtype=T.float).to(self.device)
        state = state.unsqueeze(0) #为state添加bitch size 维度
        action = self.actor.forward(state).squeeze()

        self.actor.train()
        
        if train:
            noise = T.tensor(self.ou_noise(), dtype=T.float).to(self.device)
            action = action + noise
        
        action = T.clamp(action, T.tensor(self.action_space.low).to(self.device), T.tensor(self.action_space.high).to(self.device))
        action = action.detach().cpu().numpy()

        return action.reshape((1,)).astype(np.float32)  # 转换为 (1,) 形状的数组

    
 
    def learn(self):
        if not self.memory.ready():
            return
        
        ################### KAN 网络训练 #########################################################
        # dataset = {
        # 'train_input': T.tensor(self.memory.state_memory, dtype=T.float).to(device),
        # 'train_label': T.tensor(self.memory.action_memory, dtype=T.float).to(device),  # 修改键名
        # 'test_input': T.tensor(self.memory.state_memory, dtype=T.float).to(device),
        # 'test_label': T.tensor(self.memory.action_memory, dtype=T.float).to(device)    # 修改键名
        # }
        # self.train_kan(dataset)  # 在DDPG训练过程中使用KAN优化
        ############################################################################################

        self.global_step += 1

        states, actions, reward, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(reward, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            q_ = self.target_critic.forward(next_states_tensor, next_actions_tensor).view(-1)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_
        q = self.critic.forward(states_tensor, actions_tensor).view(-1)

        ##更新critic网络
        critic_loss = F.mse_loss(q, target)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()

        # 裁剪梯度
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)  

        # 计算裁剪后的梯度范数
        # critic_gradient_norm = 0
        # for p in self.critic.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.detach().data.norm(2)
        #         critic_gradient_norm += param_norm.item() ** 2
        # critic_gradient_norm = critic_gradient_norm ** 0.5
        # print("Critic Gradient Norm (after clipping):", critic_gradient_norm)
        self.critic.optimizer.step()

        new_actions_tensor = self.actor.forward(states_tensor)
        actor_loss = -T.mean(self.critic.forward(states_tensor, new_actions_tensor)) 

        ##更新actor网络
        self.actor.optimizer.zero_grad()
        actor_loss.backward()

        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        # 计算裁剪后的梯度范数
        # actor_gradient_norm = 0
        # for p in self.actor.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.detach().data.norm(2)
        #         actor_gradient_norm += param_norm.item() ** 2
        # actor_gradient_norm = actor_gradient_norm ** 0.5
        # print("Actor Gradient Norm (without clipping):", actor_gradient_norm)
        self.actor.optimizer.step()

        # 写入裁剪后的梯度和loss值
        # self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.global_step)
        # self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.global_step)
        # self.writer.add_scalar("Gradient Norm/Critic", critic_gradient_norm, self.global_step)
        # self.writer.add_scalar("Gradient Norm/Actor", actor_gradient_norm, self.global_step)

        self.update_network_parameters()

    def train_kan(self, dataset):
        # 使用 KAN 网络的 fit 函数来训练 actor 和 critic
        self.actor.mu.fit(
            dataset=dataset,
            opt="Adam",
            steps=40,
            lamb=0.02,
            save_fig=True,
            img_folder='KAN_picture1'
        )

        self.critic.q_network.fit(
            dataset=dataset,
            opt="Adam",
            steps=20,
            lamb=0.01,
            save_fig=True,
            img_folder='KAN_picture2'
        )
 
    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic.save_checkpoint(self.checkpoint_dir + 'Critic/DDPG_critic_{}'.format(episode))
        print('Saving critic network successfully!')
        self.target_critic.save_checkpoint(self.checkpoint_dir +
                                           'Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Saving target critic network successfully!')
 
    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                        'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic.load_checkpoint(self.checkpoint_dir + 'Critic/DDPG_critic_{}'.format(episode))
        print('Loading critic network successfully!')
        self.target_critic.load_checkpoint(self.checkpoint_dir +
                                        'Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Loading target critic network successfully!')
