import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from network import ActorCriticNetwork

#Agent : Constructor, choose action, reward memory : 생성자, 행동 선택, 보상 기억
#Gamma, policy network, reward의 기억목록, log probs : 감마, policy network, 보상의 기억목록, 확률 로그
#PyTorch : categorical distribution for action selection : 파이토치에서 행동 선택에 따른 category쪽의 분포
class Agent():
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4, fc1_dims =256, fc2_dims =256, chkpt_dir='models/'):
        self.gamma = gamma
        self.action = None
        self.chkpt_dir = chkpt_dir
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))
        
    def save_models(self):
        self.actor_critic.save(self.chkpt_dir + 'actor_critic')
        print('model saved successfully')
        
    def load_models(self):
        self.actor_critic = keras.models.load_model(self.chkpt_dir +'actor_critic')
        print('model loaded successfully')
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probabilities = self.actor_critic(state)
        action_probs = tfp.distributions.Categorical(probs=probabilities)
        action = action_probs.sample()
        self.action = action
        return action.numpy()[0]
    

    def learn(self, state, reward, state_, done):
        
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
            
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)
            delta = reward + self.gamma *state_value_ *(1-int(done))-state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
            
        params = self.actor_critic.trainable_variables
        grads = tape.gradient(loss, params)
        self.actor_critic.optimizer.apply_gradients(zip(grads, params))
        