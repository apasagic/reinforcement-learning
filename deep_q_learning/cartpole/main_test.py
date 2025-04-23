import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
# Optional: suppress general TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = warnings only, '3' = errors only

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(utils_path)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import epsilon_greedy_policy
from agent import QTableAgent
from replay_buffer import ReplayBuffer

import time

# Action map: w = up, a = left, s = down, d = right
#action_map = {
#    'w': 0,  # Up
#    'a': 1,  # Left
#    's': 2,  # Down
#    'd': 3   # Right
#}

alpha = 0.01  # Learning rate;
gamma = 0.95

# Parameters
epsilon_start = 1.0
epsilon_end = 0.05
decay_rate = 0.99
num_steps = 500
num_episodes = 500 # Number of episodes to train the agent

epsilon = epsilon_start

#env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human");
env = gym.make("CartPole-v1")  # CartPole-v1: 4 (position, velocity, angle, angular velocity)
env = gym.wrappers.RecordEpisodeStatistics(env)
env.reset()
env.render()

ind_step = 0;
ind_episode = 0;

done = False;
truncated = False;

state_space_size = env.observation_space._shape[0]  # CartPole-v1: 4 (position, velocity, angle, angular velocity)
action_space_size = env.action_space.n      # CartPole-v1: 2 (left, right)

# Number of steps after which to update Q-values
no_step_update = 5 
batch_size = 64
step_count = 0

#Q = np.zeros((state_space_size, action_space_size))  # Q[s, a]

q_table_agent = QTableAgent(
    state_space_size=state_space_size,
    action_space_size=action_space_size,
    alpha=alpha,
    gamma=gamma,
    eps_start=epsilon_start,
    eps_end=epsilon_end,
    decay_rate=decay_rate,
    batch_size=batch_size
)

for ind_episode in range(num_episodes):
  
  state = env.reset()[0]
  done = False
 
  #print("Episode number: ", ind_episode)
  #print("Episode number: ", ind_episode, "Epsilon: ", q_table_agent.epsilon)

  for ind_step in range(num_steps):

    action = q_table_agent.get_action(state, env)  # Choose action with epsilon-greedy policy

    state_new, reward, done, truncated, info = env.step(action)
    
    q_table_agent.store_experience(state, action, reward, state_new, done)

    if(step_count%no_step_update == 0 or done or truncated):
      # Update Q-values using the Bellman equation
      q_table_agent.update_q(state, action, reward, state_new)
    
    state=state_new;
    step_count+=1

    env.render()
    if done or truncated:
      q_table_agent.increment_counter(reward)
      
      if "episode" in info:
        ep_info = info["episode"]
        print(f"Episode {ind_episode + 1}: reward = {ep_info['r']}, length = {ep_info['l']}")
      break
      
  q_table_agent.update_epsilon()  # Update epsilon after each episode	


print(q_table_agent.Q)
print("Successes: ", q_table_agent.successes)
env.close()

# Save the model architecture and weights
model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'dqn_model.h5'))
q_table_agent.QNetwork.save(model_save_path)
print(f"Model saved to {model_save_path}")