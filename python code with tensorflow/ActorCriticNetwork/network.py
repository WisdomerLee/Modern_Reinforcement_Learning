import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class ActorCriticNetwork(keras.Model):
    def __init__(self, lr, n_actions, fc1_dims =1024, fc2_dims = 512):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')
        self.v = Dense(1, activation=None)
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        pi = self.pi(x)
        v = self.v(x)
        
        return v,pi
    

    