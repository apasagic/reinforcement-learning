import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

#gym.wrappers.FrameStackObservation

import sys
import os
import time

# Optional: suppress general TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = warnings only, '3' = errors only

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(utils_path)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import epsilon_greedy_policy
from utils import moving_average
from agent import DeepQAgent
from replay_buffer import ReplayBuffer
import environment

# Action map: w = up, a = left, s = down, d = right
#action_map = {
#    'w': 0,  # Up
#    'a': 1,  # Left
#    's': 2,  # Down
#    'd': 3   # Right
#}

alpha = 0.0005  # Learning rate;
gamma = 0.95

# Parameters
epsilon_start = 0.85
epsilon_end = 0.05
decay_rate = 0.9995
num_steps = 10000
num_episodes = 10000 # Number of episodes to train the agent
min_sample = 200

epsilon = epsilon_start

env = environment.make_env()
env.reset()
env = gym.wrappers.RecordEpisodeStatistics(env)
#env.reset()
#env.render()

ind_step = 0;
ind_episode = 0;

done = False;
truncated = False;

state_space_size = env.observation_space._shape[0]
action_space_size = env.action_space.n

# Number of steps after which to update Q-values
no_step_train = 5
no_step_update_target = 5000
batch_size = 32

episode_rewards = []
episode_lengths = []

#Q = np.zeros((state_space_size, action_space_size))  # Q[s, a]

DQN_agent = DeepQAgent(
    state_space_size=state_space_size,
    action_space_size=action_space_size,
    alpha=alpha,
    gamma=gamma,
    eps_start=epsilon_start,
    eps_end=epsilon_end,
    decay_rate=decay_rate,
    batch_size=batch_size,
    min_sample = min_sample,
    no_step_update_target = no_step_update_target,
    no_step_train = no_step_train,
    load_checkpoint = True,
    load_deque = True
)

for ind_episode in range(num_episodes):

  state = env.reset()[0]
  done = False

  state = np.array(state, dtype=np.float32) / 255.0
  state = np.transpose(state, (1, 2, 0))

  for ind_step in range(num_steps):

    action = DQN_agent.get_action(state, env)  # Choose action with epsilon-greedy policy

    state_new, reward, done, truncated, info = env.step(action)
    state_new = np.array(state_new, dtype=np.float32) / 255.0
    state_new = np.transpose(state_new, (1, 2, 0))

    DQN_agent.store_experience(state, action, reward, state_new, done)

    if(DQN_agent.no_steps%no_step_train == 0):
      # Update Q-values using the Bellman equation
      DQN_agent.update_q(state, action, reward, state_new)

    state=state_new;
    DQN_agent.no_steps+=1

    #env.render()
    if done or truncated:
      DQN_agent.increment_counter(reward)

      if "episode" in info:
        ep_info = info["episode"]
        episode_rewards.append(ep_info['r'])
        episode_lengths.append(ep_info['l'])
        print(f"Episode {ind_episode + 1}: reward = {ep_info['r']}, length = {ep_info['l']}, epsilon = {DQN_agent.epsilon}")
      break

  DQN_agent.update_epsilon()  # Update epsilon after each episode


# Plot total reward
plt.figure(figsize=(12, 5))
plt.plot(episode_rewards, label='Reward per episode')
plt.plot(moving_average(episode_rewards), label='Moving average (10 episodes)', color='orange')
plt.plot(episode_lengths, label='Episode Length')
plt.plot(moving_average(episode_lengths), label='Moving Avg (10)', color='green')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress: Reward per Episode')
plt.legend()
plt.grid()
plt.show()

#print(DQN_agent.Q)
#print("Successes: ", DQN_agent.successes)
env.close()

# Save the model architecture and weights
model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'dqn_breakout_model.h5'))
q_table_agent.QNetwork.save(model_save_path)
print(f"Model saved to {model_save_path}")