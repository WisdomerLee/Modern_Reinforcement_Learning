
import numpy as np

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prov + self.theta*(self.mu - self.x_prev) * self.dt + self.sigma *np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        

    
#Replay Buffer : Fix size, overwrite early memory
#샘플 memories : 모두 동일하게
#__init__()
#store_transition()
#generic > no casting to pytorch tensors
#아래의 내용은 buffer.py로 별도의 스크립트로 분리하기

import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones
    

#Critic Network
#2개의 hidden layers : 400x300 : ReLU activation 함수 사용
#Adam optimizer 1 x 10^-3; L2 weight decay 0.01
#Output layer random weights [-3x10^-3, 3x10^-3]
#Other layers rnadom weights[-1/sqrt(f), 1/sqrt(f)]
#Batch normalization prior to action input(2nd HL)
#Save load checkpoint
#network.py로 저장
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        
        self.q = nn.Linear(self.fc2_dims, 1)
        
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        
        f2 = 1./np.sqrt(self.fc2.wieght.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc.bias.data.uniform_(-f2, f2)
        
        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        
        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)
        
        self.optimizer = optim.Adam(self.parameters(), lr= beta, weight_dacay=0.01)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        
        return state_action_value
    
    def save_checkpoint(self):
        print('...saving checkpoint ....')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('...loading checkpoint ....')
        self.load_checkpoint(T.load(self.checkpoint_file))
        
#Actor Network
#two hidden layer : 400x300, ReLU & tanh activation
#Adam optimizer 1x10^-4
#output layer : random weights[-3x10^-3, 3x10^-3]
#other layers : random weights[-1/sqrt(f), 1/sqrt(f)]
#Batch normalization all layers
#save load check point

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')
        
        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)
        
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        
        f2 = 1./np.sqrt(self.fc2.wieght.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc.bias.data.uniform_(-f2, f2)
        
        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        
        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)
        
        self.optimizer = optim.Adam(self.parameters(), lr= beta, weight_dacay=0.01)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))
        
        return x
    
    def save_checkpoint(self):
        print('... saving checkpoint ....')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ....')
        self.load_state_dict(T.load(self.checkpoint_file))
        

#DDPG: 
#Agent Initializer
#__init__, choose_action, store_transition
#Actor, critic network, target network
#hyperparameters
#replay memory, action noise functionality
#save checkpoint
#

import numpy as np
import torch as T
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from buffer import ReplayBuffer

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.09, max_size=10000000, fc1_dims=400, fc2_dims=300, batch_size =64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        
        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='actor')
        
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='critic')
        
        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='target actor')
        
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='target critic')
        
        self.update_network_parameters(tau=1)
        
    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        
        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        

#Agent Learn
#sample memory uniformly
#update critic by minimizing
#update actor(pytorch handle chain rule)
#value(q) of terminal state is 0
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards, states, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        
        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)
        
        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)
        
        target = rewards + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)
        
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
#Network copy
#soft update rule
#get named parameters, copy to dict, modify, upload
#initial values in __init__
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()
            
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
        
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)

#Network copy
#play 1000 games, reset noise each game
#plot learning curve, 

import gym
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape, tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    
    best_score = env.reward_range[0]
    score_history=[]
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score= avg_score
            agent.save_models()
        print('episode', i, 'score %.1f' % score, 'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
    