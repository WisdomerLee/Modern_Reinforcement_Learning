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
    def __init__(self, alpha=0.0001, beta =0.001, gamma=0.99, n_actions=4, fc1_dims =400, fc2_dims =300,
                 max_size=1_000_000, tau=0.005, input_dims= [8], batch_size = 100, noise=0.1,warmup=1000, chkpt_dir='models/', env=None, update_actor_interval=2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.chkpt_dir = chkpt_dir
        self.update_actor_iter = update_actor_interval
        self.warmup = warmup
        self.time_step = 0
        self.learn_step_cntr = 0
        
        
        self.actor = ActorNetwork(n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic_1 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic_2 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_actor = ActorNetwork(n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_critic_1 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_critic_2 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))
        
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
            
        weights=[]
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau+targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights=[]
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight*tau+targets[i]*(1-tau))
        self.target_critic_1.set_weights(weights)
        
        weights=[]
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight*tau+targets[i]*(1-tau))
        self.target_critic_2.set_weights(weights)

    def save_models(self):
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic_1.save(self.chkpt_dir + 'critic_1')
        self.critic_2.save(self.chkpt_dir + 'critic_2')
        self.target_actor.save(self.chkpt_dir + 'target_actor')
        self.target_critic_1.save(self.chkpt_dir + 'target_critic_1')
        self.target_critic_2.save(self.chkpt_dir + 'target_critic_2')
        print('model saved successfully')
        
    def load_models(self):
        self.actor = keras.models.load_model(self.chkpt_dir +'actor')
        self.critic_1 = keras.models.load_model(self.chkpt_dir +'critic_1')
        self.critic_2 = keras.models.load_model(self.chkpt_dir +'critic_2')
        self.target_actor = keras.models.load_model(self.chkpt_dir +'target_actor')
        self.target_critic_1 = keras.models.load_model(self.chkpt_dir +'target_critic_1')
        self.target_critic_2 = keras.models.load_model(self.chkpt_dir +'target_critic_2')
        print('model loaded successfully')
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
        
    def choose_action(self, observation):
        if self.time_step <self.warmup:
            mu = np.random.normal(scale= self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation])
            mu = self.actor(state)[0]
        mu_prime = mu + np.random.normal(scale= self.noise)
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
        self.time_step +=1
        
        return mu_prime
    
    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, dones = self.memory.sample_buffer(self.batch_size)
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic((state, new_policy_actions))
            actor_loss = tf.math.reduce_mean(actor_loss)
            
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))
        
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)
            q1_ = self.target_critic_1((states_, target_actions))
            q2_ = self.target_critic_2((states_, target_actions))
            
            q1 = tf.squeeze(self.critic_1((states, actions)), 1)
            q2 = tf.squeeze(self.critic_2((states, actions)), 1)
            
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)
            
            critic_value = tf.math.minimum(q1_, q2_)
            target = rewards + self.gamma *critic_value_ *(1-dones)
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)
        
        params_1 = self.critic_1.trainable_variables
        params_2 = self.critic_2.trainable_variables
        grads_1 = tape.gradient(critic_1_loss, params_1)
        grads_2 = tape.gradient(critic_2_loss, params_2)
        
        self.critic_1.optimizer.apply_gradients(zip(grads_1, params_1))
        self.critic_2.optimizer.apply_gradients(zip(grads_2, params_2))
        
        self.learn_step_cntr += 1        

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1((states, new_actions))
            actor_loss = -tf.math.reduce_mean(critic_1_value)
            
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.critic.optimizer.apply_gradients(zip(grads, params))
        
        self.update_network_parameters()
        