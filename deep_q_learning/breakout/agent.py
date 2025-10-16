import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from replay_buffer import ReplayBuffer
from collections import deque

import json
from pathlib import Path

import gymnasium as gym
import ale_py
# if using gymnasium
#import shimmy

import gymnasium as gym # or "import gymnasium as gym"
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import AtariPreprocessing

import numpy as np
import matplotlib.pyplot as plt

#   from tensorflow.keras import layers
import time

# Optional: suppress general TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = warnings only, '3' = errors only

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(utils_path)

from utils import epsilon_greedy_policy  # Only if you're using it in the class
from replay_buffer import ReplayBuffer

class DeepQAgent:
    def __init__(self, state_space_size, action_space_size, alpha, gamma, eps_start, eps_end, decay_rate, batch_size, min_sample, no_step_update_target, no_step_train, load_checkpoint,load_deque):

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = eps_start
        self.epsilon_end = eps_end
        self.epsilon = eps_start
        self.decay_rate = decay_rate  # decay factor per episode or step
        self.successes = 0
        self.success_average = 0
        self.Q = np.zeros((state_space_size, action_space_size))  # Q[s, a]
        self.Q_next = np.zeros((state_space_size, action_space_size))  # Q[s, a]
        self.count = 1
        self.batch_size = batch_size  # Batch size for training
        self.load_checkpoint = load_checkpoint
        self.min_sample = min_sample
        self.no_step_update_target = no_step_update_target
        self.no_step_train = no_step_train
        self.no_steps = 0
        self.load_deque = load_deque  # Set to True if you want to load the replay buffer from a file
        self.buffer = ReplayBuffer()
        self.q_buffer = deque(maxlen=10000)  # Buffer to store average Q values for monitoring

        # Create a figure for Q value monitoring
        fig_qval, (ax_qavr, ax_qmax) = plt.subplots(2, 1, figsize=(10, 4))  # 1 row, 2 columns

        # Example usage of subplot handles
        ax_qavr.set_title("Average Q Values")
        ax_qavr.set_xlabel("Episodes")
        ax_qavr.grid()
        ax_qmax.set_title("Q value distribution per action")
        ax_qmax.set_xlabel("Actions")
        ax_qmax.grid()

        plt.tight_layout()  # Optional: better spacing
        plt.show()
        plt.ion()

        self.ax_qavr = ax_qavr
        self.ax_qmax = ax_qmax
        self.fig_qval = fig_qval

        self.QNetwork = tf.keras.Sequential([
            #layers.Dense(24, activation='relu', input_shape=(state_space_size,)),
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(action_space_size, activation='linear')
        ])

        self.TargetNet = tf.keras.Sequential([
            #layers.Dense(24, activation='relu', input_shape=(state_space_size,)),
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(action_space_size, activation='linear')
        ])

        #self.weights_path = f"../../logs/QAgent/model_weights_DQN.weights.h5"
        #self.model_path = f"../../logs/QAgent/DQN_QNetwork.keras"

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(BASE_DIR, "..", "..", "logs", "QAgent", "model_weights_DQN.weights.h5")
        self.model_path = os.path.join(BASE_DIR, "..", "..", "logs", "QAgent", "DQN_QNetwork.keras")

        # TENSOR BOARD
        #current_time = time.strftime("%Y%m%d-%H%M%S")
        #self.log_dir = ...
        #self.tensorboard_writer = tf.summary.create_file_writer(self.log_dir)

        # TENSOR BOARD
        #self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #    log_dir=self.log_dir,
        #    histogram_freq=1,
        #    write_graph=True,
        #    write_images=False,
        #)

        #self.QNetwork.build()
        #self.TargetNet.build()

        self.QNetwork.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha, clipnorm=1.0), loss='mse')
        self.TargetNet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha, clipnorm=1.0), loss='mse')

        if self.load_checkpoint:
            self.QNetwork.load_weights(self.weights_path)
        else:
            # Save entire model
            self.QNetwork.save(self.model_path)

        #path_deque = f"../../checkpoint.pkl.gz"
        path_deque = os.path.join(BASE_DIR, "..", "..", "checkpoint.pkl.gz")

        #Load deque content if Enabled
        if self.load_deque:
            epsilon, steps_done, episode_number = self.buffer.load_checkpoint_compressed(path_deque)
            self.epsilon = epsilon
            self.no_steps = steps_done
            self.count = episode_number
            print(f"Replay buffer loaded with {len(self.buffer)} samples.")
         
        #self.QNetwork.load_weights("C:/Users/a.pasagic/Python Projects/Reinforcement-learning/reinforcement-learning/models/qnetwork_weights.weights.h5")
        # Initialize the target network with the same weights as the Q-network
        self.TargetNet.set_weights(self.QNetwork.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)
        return self.epsilon

    def get_action(self, state, env):
        return epsilon_greedy_policy(state, self.QNetwork, self.epsilon, env)

    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update_q(self, state, action, reward, next_state):

        # Update Q-values using the Bellman equation
        #if len(self.buffer) < self.batch_size:
        if len(self.buffer) < self.min_sample:
            return  # Not enough samples

        batch = self.buffer.sample(self.batch_size)

        # Vectorized version using NumPy
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Predict Q-values for states and next states
        q_values = self.QNetwork.predict(states,verbose=0)
        q_next = self.TargetNet.predict(next_states,verbose=0)

        # Calculate the target Q-values
        target_q = q_values.copy()
        target_q[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # Fit model on the batch
        self.QNetwork.fit(states, target_q, epochs=1, verbose=0)
        #self.QNetwork.fit(states, target_q, epochs=1, verbose=0, callbacks=[self.tensorboard_callback])

        # Update the target network every few episodes  (e.g., every 10 episodes)
        if self.no_steps % self.no_step_update_target == 0:
            self.TargetNet.set_weights(self.QNetwork.get_weights())
            # Save weights
            self.QNetwork.save_weights(self.weights_path)
            print(f"Target network updated after {self.no_steps} steps")

        #

        self.q_values = q_values
        return q_values

    def increment_counter(self,reward):
        
        self.count = self.count +1;

        # Clear previous plots to avoid overplotting
        self.ax_qavr.clear()
        self.ax_qmax.clear()

        if(len(self.q_buffer)>10):
            q_avr = list(self.q_buffer)

            #Update the Q value buffer for monitoring and plot the updated function
            self.q_buffer.append(np.average(self.q_values))
            qavr_mean = np.convolve(q_avr, np.ones(10)/10, mode='valid')
            self.ax_qavr.plot(np.arange(len(qavr_mean)),qavr_mean)
            self.ax_qmax.bar(np.arange(4),np.bincount(np.argmax(self.q_values,axis=1),minlength=4))
            self.fig_qval.canvas.draw()
            self.fig_qval.canvas.flush_events()  # Plot updates, BUT program continues

          #with self.tensorboard_writer.as_default():,
          #  tf.summary.scalar("epsilon", self.epsilon, step=s  elf.count)
          #  tf.summary.scalar("success_rate", self.success_average, step=self.count)
          #  self.tensorboard_writer.flush()

          #print(f"Episode: {self.count}, Success rate: {reward:.3f}, Epsilon: {self.epsilon:.6f}")

