import torch as T
import torch.nn.functional as F
import numpy as np
from noise import OUActionNoise
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
import os
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

##检查nan值函数
def check_nan(tensor, name):
    if T.isnan(tensor).any():
        print(f"NaN detected in {name}")

class TD3:
    def __init__(self, alpha, beta, state_dim, action_dim,  observation_space, action_space, ckpt_dir, gamma=0.99, tau=0.005, action_noise=0.3,
                 policy_noise=0.2, policy_noise_clip=0.2, delay_time=2, max_size=131072,batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.ou_noise = OUActionNoise(mu=np.zeros(action_dim), sigma=action_noise, decay=0.99)##加入噪声衰减
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir

        self.actor = ActorNetwork(observation_space, action_space, alpha)
        self.critic1 = CriticNetwork(observation_space, action_space, beta)
        self.critic2 = CriticNetwork(observation_space, action_space, beta)

        self.target_actor = ActorNetwork(observation_space, action_space, alpha)
        self.target_critic1 = CriticNetwork(observation_space, action_space, beta)
        self.target_critic2 = CriticNetwork(observation_space, action_space, beta)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=self.tau)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, train=True):
        self.actor.eval()
        state = T.tensor(np.array(observation, dtype=np.float32), dtype=T.float).to(device)
        state = state.unsqueeze(0) #为state添加bitch size 维度
        action = self.actor.forward(state).squeeze()
        self.actor.train()
        if train:
            # exploration noise
            # noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),dtype=T.float).to(device)
            ##使用ou噪声##
            noise = T.tensor(self.ou_noise(),dtype=T.float).to(device)
            action = action + noise
            action = T.clamp(action,T.tensor(self.action_space.low).to(device),T.tensor(self.action_space.high).to(device))
        action = action.detach().cpu().numpy()  
        return action.reshape((1,)).astype(np.float32) # 转换为 (1,) 形状的数组

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            ############################################################
            #高斯噪声
            # action_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),dtype=T.float).to(device)
            # action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            # next_actions_tensor = T.clamp(next_actions_tensor+action_noise, -1, 1)
            #############################################################
            # 使用 OU 噪声替代高斯噪声
            policy_noise = T.tensor(self.ou_noise(), dtype=T.float).to(device)
            policy_noise = T.clamp(policy_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = T.clamp(next_actions_tensor + policy_noise, -1, 1)
            #############################################################
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            critic_val = T.min(q1_, q2_)
            target = rewards_tensor + self.gamma * critic_val
        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return

        new_actions_tensor = self.actor.forward(states_tensor)
        q1 = self.critic1.forward(states_tensor, new_actions_tensor)
        actor_loss = -T.mean(q1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic1.save_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Saving critic1 network successfully!')
        self.target_critic1.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Saving target critic1 network successfully!')
        self.critic2.save_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Saving critic2 network successfully!')
        self.target_critic2.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Saving target critic2 network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic1.load_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Loading critic1 network successfully!')
        self.target_critic1.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Loading target critic1 network successfully!')
        self.critic2.load_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Loading critic2 network successfully!')
        self.target_critic2.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Loading target critic2 network successfully!')
