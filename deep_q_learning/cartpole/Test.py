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
from utils import epsilon_greedy_policy, print_policy_from_q, plot_policy_and_value

# Get the path to the models folder (two levels up from the script)
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "models"))
print(models_dir)
# Check and print
print(f"Looking in: {models_dir}")

if os.path.exists(models_dir):
    print("Contents of 'models' folder:")
    print(os.listdir(models_dir))
else:
    print("The 'models' folder does not exist at the expected location.")

alpha = 0.1;
gamma = 0.95;

# Parameters
epsilon_start = 0.1
epsilon_end = 0.1
decay_rate = 0.99999  # decay factor per episode or step

epsilon = epsilon_start

#env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human");
env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False,render_mode="human");
env.reset()
env.render()  

num_steps = 25
num_episodes = 1 # Number of episodes to train the agent

ind_step = 0;
ind_episode = 0;

done = False;
truncated = False;

V = np.zeros(env.observation_space.n)  # Value function
#policy = np.zeros(env.observation_space.n)  # Policy

state_space_size = env.observation_space.n  # 16 for 4x4 map
action_space_size = env.action_space.n      # 4 (Left, Down, Right, Up)

file_path = r"C:\Users\a.pasagic\Python Projects\Reinforcement-learning\reinforcement-learning\models\q_table.npy"

Q = np.load(file_path)

print("Q-table loaded successfully.")

successes = 0;
success_average = 0;
count = 0;

for ind_episode in range(num_episodes):
  
  state = env.reset()[0]
  done = False
 
  #print("Episode number: ", ind_episode)

  for ind_step in range(num_steps):

    #action = np.argmax(Q[state]) # Choose action with highest Q-value (greedy policy)
    action = epsilon_greedy_policy(state, Q, epsilon, env)
    state, reward, done, truncated, info = env.step(action)

print_policy_from_q(Q, env)
V = np.max(Q, axis=1)
plot_policy_and_value(Q, V, env)

env.close()

