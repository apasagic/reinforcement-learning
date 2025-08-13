# DEFINE AND TEST ENVIRONMENT
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

# -------------------------------------
# 👇 LifePenaltyWrapper definition
# -------------------------------------
class LifePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, life_loss_penalty=-1.0):
        super().__init__(env)
        self.life_loss_penalty = life_loss_penalty
        self.prev_lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = self.env.unwrapped.ale.lives()

        if current_lives < self.prev_lives and current_lives > 0:
            reward += self.life_loss_penalty
            info["life_lost_penalty"] = self.life_loss_penalty

        self.prev_lives = current_lives
        return obs, reward, terminated, truncated, info

# -------------------------------------
# 👇 Environment builder
# -------------------------------------
def make_env():
    env_id = "BreakoutNoFrameskip-v4"
    env = gym.make(env_id, render_mode="rgb_array")
    env = LifePenaltyWrapper(env, life_loss_penalty=-0.5)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)
    return env

# -------------------------------------
# 👇 Plot stacked frames
# -------------------------------------
def plot_stacked_frames(obs, frame_index):
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for i in range(4):
        axes[i].imshow(obs[i, :, :], cmap='gray')
        axes[i].set_title(f"Frame {frame_index}-{i+1}")
        axes[i].axis("off")
    plt.suptitle("💀 Life lost! Frame stack")
    plt.tight_layout()
    plt.show()

# -------------------------------------
# 👇 Test block
# -------------------------------------
if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()

    for i in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if "life_lost_penalty" in info:
            print(f"Step {i}: 💀 Life lost! Penalty = {info['life_lost_penalty']}")
            plot_stacked_frames(obs, i)

        if terminated or truncated:
            obs, info = env.reset()

    