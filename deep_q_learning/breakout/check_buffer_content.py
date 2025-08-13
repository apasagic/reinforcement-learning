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

