from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium_qr.envs.basketball_shooter import BasketballShooterEnv


class NoBasket(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not BasketballShooterEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the BasketballShooterEnv class.')

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        env = self.env.unwrapped

        render_mode = env.render_mode
        env.render_mode = None

        observation, info = super().reset(seed=seed, options=options)

        basket = env._basket
        for fixture in basket.fixtures:
            basket.DestroyFixture(fixture)

        env.render_mode = render_mode
        if env.render_mode == "human":
            env._render_frame()

        return observation, info

    def step(self, action: np.ndarray = None):
        return super().step(action)
