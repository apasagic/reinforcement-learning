# BUFFER DEFINITION
import numpy as np
import copy

from collections import deque
import random

import gzip
import pickle

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):

        # Convert images to uint8 to save memory
        #state = np.array(state)
        #next_state = np.array(next_state)
        #action = np.int32(action)
        #reward = np.float16(reward)
        #done = np.bool_(done)

        #self.buffer.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done))
        self.buffer.append(copy.deepcopy((state, action, reward, next_state, done)))

    def sample(self, batch_size=32):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

# Load state
    def load_checkpoint_compressed(self,filename):
        with gzip.open(filename, 'rb') as f:
           checkpoint_data = pickle.load(f)

        self.buffer = deque(checkpoint_data['replay_buffer'], self.capacity)
        epsilon = checkpoint_data['epsilon']
        steps_done = checkpoint_data['steps_done']
        episode_number = checkpoint_data['episode']
    
        return epsilon, steps_done, episode_number
