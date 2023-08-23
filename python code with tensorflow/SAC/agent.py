from SAC.network import ValueNetwork
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from buffer.replaybuffer import ReplayBuffer
from network import ActorNetwork, CriticNetwork

#Agent : Constructor, choose action, reward memory : 생성자, 행동 선택, 보상 기억
#Gamma, policy network, reward의 기억목록, log probs : 감마, policy network, 보상의 기억목록, 확률 로그
#PyTorch : categorical distribution for action selection : 파이토치에서 행동 선택에 따른 category쪽의 분포
class Agent():
    def __init__(self, alpha=0.0001, beta =0.001, gamma=0.99, n_actions=4, fc1_dims =256, fc2_dims =256,
                 max_size=1_000_000, tau=0.005, input_dims= [8], batch_size = 256, chkpt_dir='models/', env=None, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.chkpt_dir = chkpt_dir
        self.reward_scale = reward_scale
        
        self.actor = ActorNetwork(n_actions=n_actions, max_action=self.max_action, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic_1 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic_2 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.value = ValueNetwork(fc1_dims = fc1_dims, fc2_dims=fc2_dims)
        self.target_value = ValueNetwork(fc1_dims = fc1_dims, fc2_dims=fc2_dims)
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))
        
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
            
        weights=[]
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight*tau+targets[i]*(1-tau))
        self.target_value.set_weights(weights)
        
    def save_models(self):
        if self.memory.mem_cntr > self.batch_size:
            self.actor.save(self.chkpt_dir + 'actor')
            self.critic_1.save(self.chkpt_dir + 'critic_1')
            self.critic_2.save(self.chkpt_dir + 'critic_2')
            self.value.save(self.chkpt_dir + 'value')
            self.target_value.save(self.chkpt_dir + 'target_value')
            
            print('model saved successfully')
        
    def load_models(self):
        self.actor = keras.models.load_model(self.chkpt_dir +'actor')
        self.critic_1 = keras.models.load_model(self.chkpt_dir +'critic_1')
        self.critic_2 = keras.models.load_model(self.chkpt_dir +'critic_2')
        self.value = keras.models.load_model(self.chkpt_dir +'value')
        self.target_value = keras.models.load_model(self.chkpt_dir +'target_value')
        print('model loaded successfully')
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        

    def sample_normal(self, state):
        mu, sigma = self.actor(state)
        probabilities = tfp.distributions.Normal(mu, sigma)
        actions = probabilities.sample()
        action = tf.math.tanh(actions) * self.actor.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action, 2) + self.actor.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        
        return action, log_probs
    
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.sample_normal(state)
        
        return actions[0]
    
    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, dones = self.memory.sample_buffer(self.batch_size)
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            current_polity_actions, log_probs = self.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_pi = self.critic_1((states, current_polity_actions))
            q2_new_pi = self.critic_2((states, current_polity_actions))
            critic_value = tf.squeeze(tf.math.minimum(q1_new_pi, q2_new_pi), 1)
            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
            
        params = self.value.trainable_variables
        grads = tape.gradient(value_loss, params)
        
        
        self.value.optimizer.apply_gradients(zip(grads, params))
        
        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1((states, new_policy_actions))
            q2_new_policy = self.critic_2((states, new_policy_actions))
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)
            
            
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))
        
        with tf.GradientTape(persistent=True) as tape:
            value_ = tf.squeeze(self.target_value(states_), 1)
            q_hat = self.reward_scale * rewards + self.gamma *value_ *(1-dones)
            q1_old_policy = tf.squeeze(self.critic_1((states, actions)),1)
            q2_old_policy = tf.squeeze(self.critic_2((states, actions)),1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
            
        params_1 = self.critic_1.trainable_variables
        params_2 = self.critic_2.trainable_variables
        grads_1 = tape.gradient(critic_1_loss, params_1)
        grads_2 = tape.gradient(critic_1_loss, params_2)
        self.critic_1.optimizer.apply_gradients(zip(grads_1, params_1))
        self.critic_2.optimizer.apply_gradients(zip(grads_2, params_2))
        
        self.update_network_parameters()
        