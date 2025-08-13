## THIS CLASS IS FOR BOTH Q and TARGET NETWORKS
import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from replay_buffer import ReplayBuffer


class QNetwork:
     def __init__(self, state_space_size, action_space_size, alpha=alpha, gamma=gamma):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma

        self.model = tf.keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(state_space_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(action_space_size, activation='linear')
        ])

        