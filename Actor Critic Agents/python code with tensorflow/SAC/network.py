import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims =256, fc2_dims = 256):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        
        self.q = Dense(1, activation=None)
        
    def call(self, inputs):
        state, action = inputs
        x = self.fc1(tf.concat([state,action], axis=1))
        x = self.fc2(x)
        q = self.q(x)
        
        return q
    

class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.v(x)
        return x
    
class ActorNetwork(keras.Model):
    def __init__(self, n_actions, max_action, fc1_dims =256, fc2_dims = 256):
        super(ActorNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.mu = Dense(n_actions, activation='tanh')
        self.sigma = Dense(n_actions, activation=None)
        
        self.max_action = max_action
        self.noise = 1e-6
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma
    