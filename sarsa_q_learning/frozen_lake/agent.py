import numpy as np
import sys
import os
# Optional: suppress general TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = warnings only, '3' = errors only

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(utils_path)

from utils import epsilon_greedy_policy  # Only if you're using it in the class

class QTableAgent:
    def __init__(self, state_space_size, action_space_size, alpha, gamma, eps_start, eps_end, decay_rate):
        self.alpha = alpha;
        self.gamma = gamma;
        self.epsilon_start = eps_start
        self.epsilon_end = eps_end
        self.epsilon = eps_start
        self.decay_rate = decay_rate  # decay factor per episode or step
        self.successes = 0
        self.success_average = 0
        self.Q = np.zeros((state_space_size, action_space_size))  # Q[s, a]
        self.count = 0;

    def update_epsilon(self, episode):
        self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.decay_rate ** episode))
        return self.epsilon
    
    def get_action(self, state, env):
        return epsilon_greedy_policy(state, self.Q, self.epsilon, env)
    
    def update_q(self, state, action, reward, next_state):
        delta_target = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action]
        self.Q[state, action] += self.alpha * delta_target
        return self.Q
    
    def increment_counter(self,reward):
          
          if reward==1:  # Assuming a condition like reaching the goal
            self.successes += 1
  
          self.count = self.count +1;

          if(self.count%1000==0):
            self.success_average = self.successes/1000
            #success_rate.append(self.success_average)

            print(f"Episode: {self.count}, Success rate: {self.success_average:.2f}, Epsilon: {self.epsilon:.3f}")

            self.successes = 0 
            self.success_average = 0
            self.count = 0