import gymnasium as gym


class CartPoleRewardShaping(gym.Wrapper):
    def __init__(
        self,
        env,
        center_weight=0.03,
        angle_weight=0.015,
        position_limit=2.4,
        angle_limit=0.2095,
        ramp_start_episode=300,
        ramp_end_episode=1200,
    ):
        super().__init__(env)
        self.center_weight = center_weight
        self.angle_weight = angle_weight
        self.position_limit = position_limit
        self.angle_limit = angle_limit
        self.ramp_start_episode = ramp_start_episode
        self.ramp_end_episode = ramp_end_episode
        self.episode_number = 0

    def reset(self, **kwargs):
        self.episode_number += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        cart_pos = state[0]
        pole_angle = state[2]

        shaping_scale = self._shaping_scale()
        active_center_weight = self.center_weight * shaping_scale
        active_angle_weight = self.angle_weight * shaping_scale

        center_penalty = active_center_weight * min(abs(cart_pos) / self.position_limit, 1.0)
        angle_penalty = active_angle_weight * min(abs(pole_angle) / self.angle_limit, 1.0)
        shaped_reward = reward - center_penalty - angle_penalty

        info["raw_reward"] = reward
        info["center_penalty"] = center_penalty
        info["angle_penalty"] = angle_penalty
        info["shaped_reward"] = shaped_reward
        info["reward_shaping_scale"] = shaping_scale
        info["reward_shaping_center_weight"] = active_center_weight
        info["reward_shaping_angle_weight"] = active_angle_weight

        return state, shaped_reward, done, truncated, info

    def _shaping_scale(self):
        if self.episode_number <= self.ramp_start_episode:
            return 0.0

        if self.episode_number >= self.ramp_end_episode:
            return 1.0

        ramp_length = self.ramp_end_episode - self.ramp_start_episode
        return (self.episode_number - self.ramp_start_episode) / ramp_length
