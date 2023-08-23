import pybullet_envs
import gym
import numpy as np
from agent import Agent
from PolicyGradientNetwork.utils import plot_learning_curve, manage_memory

if __name__ == '__main__':
    manage_memory()
    env = gym.make('InvertedPendulumBulletEnv-v0')
    n_games = 2000
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0], env=env)
    load_checkpoint = False
    best_score = env.reward_range[0]
    
    if load_checkpoint:
        agent.load_models()
    
    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score= 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score
            
        print('episode {} score {:.1f}average score {:.1f}'.format(i, score, avg_score))
    figure_file = 'plots/inverted_pendulm.png'
    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores,figure_file)