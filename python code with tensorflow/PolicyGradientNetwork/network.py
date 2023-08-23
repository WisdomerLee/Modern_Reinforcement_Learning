import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class PolicyNetwork(keras.Model):
    def __init__(self, lr, n_actions, fc1_dims =256, fc2_dims = 256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.pi(x)
        
        return x

    