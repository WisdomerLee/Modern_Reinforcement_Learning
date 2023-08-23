
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from network import PolicyNetwork

#Agent : Constructor, choose action, reward memory : 생성자, 행동 선택, 보상 기억
#Gamma, policy network, reward의 기억목록, log probs : 감마, policy network, 보상의 기억목록, 확률 로그
#PyTorch : categorical distribution for action selection : 파이토치에서 행동 선택에 따른 category쪽의 분포
class Agent():
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4, fc1_dims =256, fc2_dims =256, chkpt_dir='models/'):
        self.gamma = gamma
        self.state_memory=[]
        self.reward_memory=[]
        self.action_memory=[]
        self.chkpt_dir = chkpt_dir
        self.policy = PolicyNetwork(n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.policy.compile(optimizer=Adam(learning_rate=alpha))
        
    def save_models(self):
        self.policy.save(self.chkpt_dir + 'reinforce')
        print('model saved successfully')
        
    def load_models(self):
        self.policy = keras.models.load_model(self.chkpt_dir +'reinfoce')
        print('model loaded successfully')
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        probabilities = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probabilities)
        action = action_probs.sample()        
        return action.numpy()[0]
    
    def store_transition(self, observation, action, reward):
        self.staqte_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory(reward)

    def learn(self):
        
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)
        
        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
        # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}
        G = np.zeros_like(self.reward_memory)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t]= G_sum
        
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g,state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g*tf.squeeze(log_prob)
        params = self.policy.trainable_variables
        grads = tape.gradient(loss, params)
        self.policy.optimizer.apply_gradients(zip(grads, params))
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
