#Monte Carlo Prediction과 차이 : Monte Carlo Prediction은 특정 에피소드 단위로 학습이 끝나야 진행이 가능하다면
#Temporal Difference Learning : 특정 시간 간격마다 value function을 업데이트 함

from turtle import st
import numpy as np
import gym

def simple_policy(state):
    action = 0 if state <5 else 1
    return action

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    alpha=0.1
    gamma=0.99
    
    states = np.linspace(-0.2094, 0.2094, 10)
    V= {}
    for state in range(len(states)+1):
        V[state]=0
        
    for i in range(5000):
        observation = env.reset()
        done = False
        while not done:
            state = int(np.digitize(observation[2], states))
            action = simple_policy(state)
            observation_, reward, done, info = env.step(action)
            state_ = int(np.digitize(observation_[2], states))
            V[state] = V[state] + alpha*(reward + gamma*V[state_] - V[state])
            observation = observation_
            
    for state in V:
        print(state, '%.3f' % V[state])


#Q-Learning

import numpy as np

class Agent():
    def __init__(self, lr, gamma, n_actions, state_space, eps_start, eps_end, eps_dec):
        self.lr= lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.n_actions = n_actions
        
        self.state_space = state_space
        self.action_space = [i for i in range(self.n_actions)]
        
        self.Q = {}
        self.init_Q()
    def init_Q(self):
        for state in self.state_space:
            for action in self.action_space:
                self.Q[(state, action)] = 0.0
                
    def max_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in self.action_space])
        action = np.argmax(actions)
        return action
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.max_action(state)
            
        return action
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_end else self.eps_end
        
    def learn(self, state, action, reward, state_):
        a_max = self.max_action(state_)
        self.Q[(state, action)] = self.Q[(state, action)] + self.lr *(reward + self.gamma*self.Q[(state, a_max)] - self.Q[(state, action)])
        

import gym
import matplotlib.pyplot as plt
import numpy as np

class CartPoleStateDigitizer():
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins = 10):
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space= np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocty_space = np.linspace(-1*bounds[3], bounds[3], n_bins)
        self.states = self.get_state_space()
        
    def get_state_space(self):
        states = []
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocty_space)+1):
                        states.append((i,j,k,l))
        return states
    def dgitize(self, observation):
        x, x_dot, theta, theta_dot = observation
        cart_x = int(np.digitize(x, self.position_space))
        cart_xdot = int(np.digitize(x_dot, self.velocity_space))
        pole_theta = int(np.digitize(theta, self.pole_angle_space))
        pole_thetadot = int(np.digitize(theta_dot, self.pole_velocty_space))
        return (cart_x, cart_xdot, pole_theta, pole_thetadot)
    
def plot_learning_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100): (i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()
    
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    n_games = 50000
    eps_dec = 2/n_games
    digitizer = CartPoleStateDigitizer()
    agent = Agent(lr = 0.01, gamma=0.99, n_actions= 2, eps_start=1.0, eps_end=0.01, eps_dec=eps_dec, state_space=digitizer.states)
    
    scores = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        state = digitizer.digitize(observation)
        while not done:
            action = agent.choose_action(state)
            observation_, reward, done, info = env.step(action)
            state_ = digitizer.digitize(observation_)
            agent.learn(state, action, reward, state_)
            state = state_
            score += reward
        if i % 5000 ==0:
            print('episode ', i, 'score %.1f' % score, 'epsilon %.2f' % agent.epsilon)
        agent.decrement_epsilon()
        scores.append(score)
        
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(scores, x)
            
    