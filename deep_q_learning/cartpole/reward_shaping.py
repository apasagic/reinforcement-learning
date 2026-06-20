import gymnasium as gym


class CartPoleRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        center_weight=0.05,
        angle_weight=0.05,
        position_limit=2.4,
        angle_limit=0.2095,
    ):
        super().__init__(env)
        self.center_weight = center_weight
        self.angle_weight = angle_weight
        self.position_limit = position_limit
        self.angle_limit = angle_limit

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        cart_pos = state[0]
        pole_angle = state[2]

        center_penalty = self.center_weight * min(abs(cart_pos) / self.position_limit, 1.0)
        angle_penalty = self.angle_weight * min(abs(pole_angle) / self.angle_limit, 1.0)
        shaped_reward = reward - center_penalty - angle_penalty

        info["raw_reward"] = reward
        info["center_penalty"] = center_penalty
        info["angle_penalty"] = angle_penalty
        info["shaped_reward"] = shaped_reward

        return state, shaped_reward, done, truncated, info
