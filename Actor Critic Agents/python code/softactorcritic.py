#3개의 다른  network : V, Q, policy
#V: state들의 상태를 받아 scalar value값을 내놓음
#Critic : state, action을 받아 scalar value값을 내놓음
#Actor : state들을 받아 평균, sigma값을 내놓음
#constrain sigma : (0, 1)로 자르거나 / sigmoid 함수 적용
#pytorch : normal distribution : action
#reparameterization trick : sample vs rsample
#log prob : n_actions의 갯수만큼의 차원을 가짐, sum/mean에 필요
#multivariate normal distribution : 동작하지 않는 것으로 보임
#save network에서 functionality복사
#hyperparameter의 appendix 확인

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name +'_sac')
        
        #fully conncted layer
        self.fc1 = nn.Linear(self.input_dims[0]+ n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.q1 = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.f1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        
        q1= self.q1(action_value)
        return q1
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... saving checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action, n_actions, name, chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name +'_sac')
        self.reparam_noise = 1e-6
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        
        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
            
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        return action, log_probs
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... saving checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
    
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, name, chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name +'_sac')
        
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        
        v = self.v(state_value)
        return v
        
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... saving checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

#initializer, choose_action, store_transition, update_network_parameters, save_models
#hyperparameters, reward scaling
#Agent : V, Q1, Q2, Actor, target V

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, env_id, gamma=0.99, 
                 n_actions=2, max_size=100000, layer1_size=256, layer2_size=256, batch_size=100, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name=env_id +'_actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name=env_id + '_critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name=env_id + '_critic_2')
        
        self.value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, name=env_id + '_value')
        self.target_value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, name=env_id + '_target_value')
        
        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        
    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        
        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)
        
        for name in value_state_dict:
            value_state_dict[name] = tau *value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()
        
        self.target_value.load_state_dict(value_state_dict)
        
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        

#memory sampling and conversion to torch tensors
#value network update :메모리의 V
#두 critic중 최소 
#action : 현재 policy에서 고름 (memory에서 아님)
#critic update : 메모리의 scaled rewards, actions
#actor upated : reparam. trick, Q1, Q2중 최소 값
#cost.backward(retain_graph=True)
#update target value network parameters
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        
        value_[done]=0.0
        
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value = log_probs
        value_loss = F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()
        
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.step()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        
        q_hat = self.scale* reward +self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss = critic_1_loss+ critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.update_network_parameters()

#main loop
#negtive score environment : 동작 없음
#pybullet을 이용할 것
#import pybullet_envs
#import gym
#env = gym.make('InvertedPendulumBulletEnv-v0')

import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env_id = 'InvertedPendulumBulletEnv-v0'
    env = gym.make(env_id)
    agent = Agent(alpha=0.0003, bata=0.0003, reward_scale=2, env_id=env_id, 
                  input_dims=env.observation_space.shape, tau= 0.005, 
                  env=env, batch_size=256, layer1_size=256, layer2_size=256,
                  n_actions=env.action_space.shape[0])
    n_games = 250
    filename = env_id + '_' + str(n_games) + 'game_scale'+str(agent.scale)+'.png'
    figure_file = 'plots/' +filename
    
    best_score = env.reward_range[0]
    score_history=[]
    load_checkpoint= False
    
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
        
    steps = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            steps +=1
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'trailing 100 games average %.1f' % avg_score, 'steps %d' %steps, env_id, 'scale', agent.scale)
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
        
