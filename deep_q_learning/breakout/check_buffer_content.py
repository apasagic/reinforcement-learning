import random
batch_test = random.sample(self.buffer.buffer, 10000)
states_t, actions_t, rewards_t, next_states_t, dones_t = zip(*batch_test)
states_t = np.array(states_t)
actions_t = np.array(actions_t)
rewards_t = np.array(rewards_t)
next_states_t = np.array(next_states_t)
dones_t = np.array(dones_t)

q_values_t = self.QNetwork.predict(states_t,verbose=0)

print(np.bincount(np.argmax(q_values_t, axis=1), minlength=4))

import numpy as np
import matplotlib.pyplot as plt

# Suppose q_values is your (10000, num_actions) array
# Example:
# q_values = np.random.randn(10000, 4)

# Average over actions
state_values_mean = q_values_t.mean(axis=1)

# Max over actions
state_values_max = q_values_t.max(axis=1)

# Plot histograms
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(state_values_mean, bins=50, alpha=0.7)
plt.title("Distribution of Mean Q-values per State")
plt.xlabel("Mean Q-value")
plt.ylabel("Count")

plt.subplot(1,2,2)
plt.hist(state_values_max, bins=50, alpha=0.7, color='orange')
plt.title("Distribution of Max Q-values per State")
plt.xlabel("Max Q-value")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
