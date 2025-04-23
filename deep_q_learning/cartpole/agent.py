import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from replay_buffer import ReplayBuffer

#   from tensorflow.keras import layers
import time

# Optional: suppress general TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = warnings only, '3' = errors only

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(utils_path)

from utils import epsilon_greedy_policy  # Only if you're using it in the class
from replay_buffer import ReplayBuffer

class QTableAgent:
    def __init__(self, state_space_size, action_space_size, alpha, gamma, eps_start, eps_end, decay_rate, batch_size):
        
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha;
        self.gamma = gamma;
        self.epsilon_start = eps_start
        self.epsilon_end = eps_end
        self.epsilon = eps_start
        self.decay_rate = decay_rate  # decay factor per episode or step
        self.successes = 0
        self.success_average = 0
        self.Q = np.zeros((state_space_size, action_space_size))  # Q[s, a]
        self.Q_next = np.zeros((state_space_size, action_space_size))  # Q[s, a]
        self.count = 0
        self.batch_size = batch_size  # Batch size for training

        self.buffer = ReplayBuffer()

        self.QNetwork = tf.keras.Sequential([
            #layers.Dense(24, activation='relu', input_shape=(state_space_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(action_space_size, activation='linear')
        ])

        self.TargetNet = tf.keras.Sequential([
            #layers.Dense(24, activation='relu', input_shape=(state_space_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(action_space_size, activation='linear')
        ])

        # Initialize the target network with the same weights as the Q-network
        self.TargetNet.set_weights(self.QNetwork.get_weights())     

        self.QNetwork.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
        self.TargetNet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)
        return self.epsilon
    
    def get_action(self, state, env):
        return epsilon_greedy_policy(state, self.QNetwork, self.epsilon, env)
    
    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)    

    def update_q(self, state, action, reward, next_state):
        
        # Update Q-values using the Bellman equation
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples

        batch = self.buffer.sample(self.batch_size)

        # Vectorized version using NumPy
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        #start = time.time()

        # Predict Q-values for states and next states
        q_values = self.QNetwork.predict(states,verbose=0)
        q_next = self.TargetNet.predict(next_states,verbose=0)

        #end = time.time()
        #sprint(f"Prediction time: {end - start:.4f} seconds")

        # Calculate the target Q-values
        target_q = q_values.copy()
        target_q[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # Update the Q-values for the specific actions taken
        #q_values[np.arange(len(actions)), actions] = target_q

        #start = time.time()

        # Fit model on the batch
        self.QNetwork.fit(states, target_q, epochs=1, verbose=0)
        #print("Model fitting time: ", self.QNetwork.optimizer.iterations.numpy())
        #end = time.time()
        #print(f"Model fitting time: {end - start:.4f} seconds")

        # Update the target network every few steps (e.g., every 10 steps)
        if self.count % 20 == 0:
            self.TargetNet.set_weights(self.QNetwork.get_weights(),verbose=0)
            #print("Target network updated")

        return q_values
    
    def increment_counter(self,reward):
          
          if reward==1:  # Assuming a condition like reaching the goal
            self.successes += 1
  
          self.count = self.count +1;

          if(self.count%10==0):
            self.success_average = self.successes/10
            #success_rate.append(self.success_average)

            #print(f"Episode: {self.count}, Success rate: {self.success_average:.3f}, Epsilon: {self.epsilon:.6f}")

            self.successes = 0 
            self.success_average = 0
            #self.count = 0