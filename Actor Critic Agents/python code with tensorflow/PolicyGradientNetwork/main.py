import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory
    
if __name__ == '__main__':
    manage_memory()
    env = gym.make('LunarLander-v2')
    n_games = 1000
    agent = Agent(gamma=0.99, alpha=0.0005, n_actions=env.action_space.n)
    load_checkpoint = False
    best_score = -np.inf
    
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
            agent.store_transition(observation, action, reward)
            observation = observation_

        if not load_checkpoint:
            agent.learn()

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
                
            best_score = avg_score
            
        print('episode {} score {:.1f}average score {:.1f}'.format(i, score, avg_score))
    figure_file = 'plots/lunar-lander.png'
    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores,figure_file)