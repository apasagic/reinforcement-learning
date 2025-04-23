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

# Action map: w = up, a = left, s = down, d = right
#action_map = {
#    'w': 0,  # Up
#    'a': 1,  # Left
#    's': 2,  # Down
#    'd': 3   # Right
#}

alpha = 0.25;
gamma = 0.95;

# Parameters
epsilon_start = 1.0
epsilon_end = 0.05
decay_rate = 0.999999  # decay factor per episode or step

num_steps = 75
num_episodes = 5000000 # Number of episodes to train the agent

epsilon = epsilon_start

#env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human");
env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False);
env.reset()
env.render()  


ind_step = 0;
ind_episode = 0;

done = False;
truncated = False;

#V = np.zeros(env.observation_space.n)  # Value function
#policy = np.zeros(env.observation_space.n)  # Policy

state_space_size = env.observation_space.n  # 16 for 4x4 map
action_space_size = env.action_space.n      # 4 (Left, Down, Right, Up)

#Q = np.zeros((state_space_size, action_space_size))  # Q[s, a]

q_table_agent = QTableAgent(
    state_space_size=state_space_size,
    action_space_size=action_space_size,
    alpha=alpha,
    gamma=gamma,
    eps_start=epsilon_start,
    eps_end=epsilon_end,
    decay_rate=decay_rate
)

for ind_episode in range(num_episodes):
  
  state = env.reset()[0]
  done = False
 
  #print("Episode number: ", ind_episode)

  for ind_step in range(num_steps):

    action = q_table_agent.get_action(state, env)  # Choose action with epsilon-greedy policy

    #USE GREEDY POLICY FOR TESTING
    #action = np.argmax(Q[state]) # Choose action with highest Q-value (greedy policy)

    #USE MANUAL INPUT FOR DEBUGGIN
    #action = input("Enter action (w/a/s/d): ").strip().lower()
    #action = action_map[action]

    #USE RANDOM TO TEST THE ENVIRONMENT
    #action = env.action_space.sample()  # Explore

    state_new, reward, done, truncated, info = env.step(action)
    #deltaTarget = reward + gamma * np.max(Q[state_new]) - Q[state,action]
    #Q[state,action] = Q[state,action] + alpha * deltaTarget
    
    q_table_agent.update_q(state, action, reward, state_new)
    
    state=state_new;
    
    env.render()
    if done or truncated:
      q_table_agent.increment_counter(reward)
      break
      
  q_table_agent.update_epsilon(ind_episode)


print(q_table_agent.Q)
print("Successes: ", successes)
env.close()

#plt.plot(success_rate)

# Save Q-table
#file_path = os.path.join(save_path, "q_table.npy")
file_path = r"C:\Users\a.pasagic\Python Projects\Reinforcement-learning\reinforcement-learning\models\q_table.npy"
np.save(file_path, q_table_agent.Q)

print(f"Q-table saved to: {file_path}")