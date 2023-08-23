#Critic Architecture
#(state dim + action dim, 400)
#ReLU
#(action dim + 400, 300)
#ReLU
#(300, 1)

#Actor Architecture
#(state dim, 400)
#ReLU
#(400, 300)
#ReLU
#(300, 1)
#tanh

#DDPG Network

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name +'_td3')
        
        self.fc1 = nn.Linear(self.input_dims[0]+ n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def foward(self, state, action):
        q1_action_value = self.f1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        
        q1= self.q1(q1_action_value)
        return q1
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... saving checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name +'_td3')
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        prob = T.tanh(self.mu(prob)) #if action is > +/-1 then multiply by max action
        
        return prob
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... saving checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
    

#Agent Basic Functionality
#__init__, choose_action, store_transition
#gaussian noise, mean 0 sigma 0.1, clamp
#Actor, 2 critic networks, target networks

#td3
import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000,
                 n_actions=2, max_size=100000, layer1_size=400, layer2_size=300, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr =0
        self.time_step =0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='critic_2')
        
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='target actor')
        
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='target critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='target critic_2')
        
        self.noise = noise
        self.update_network_parameters(tau=1)
        
    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions, )))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        

#Agent Learn
#Sample memory uniform
#Update Critics by minimizing
#Update actor, target networks : 2step
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done),to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        
        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])
        
        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)
        
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)
        
        q1_[done]=0.0
        q2_[done]=0.0
        
        q1_=q1_.view(-1)
        q2_=q2_.view(-1)
        
        critic_value = T.min(q1_, q2_)
        
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.learn_step_cntr +=1
        
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
        
#network copy
#soft update rule
#get named paramters, copy to dict, modify, upload
#update both critics
#set initial values in __init__
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)
        
        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau *critic_1_state_dict[name].clone() + (1-tau)*target_critic_1_state_dict[name].clone()
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau *critic_2_state_dict[name].clone() + (1-tau)*target_critic_2_state_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
        
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        
#main loop
#main.py
import gym
import numpy as np
from td3_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape, tau =0.005, env= env, batch_size=100, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0])
    n_games = 1500
    filename = 'Walker2d_' + str(n_games) + '.png'
    figure_file = 'plots/'+ filename
    
    best_score=env.reward_range[0]
    score_history=[]
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
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
            best_score = avg_score
            agent.save_models()
        print('episode', i, 'score %.2f' % score, 'trailing 100 games ave %.3f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score, figure_file)
    