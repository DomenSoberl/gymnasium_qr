from typing import Optional
import copy
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium_qr.envs.basketball_shooter import BasketballShooterEnv


class OneShotScenario(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)
        self._env = env

        if type(env.unwrapped) is not BasketballShooterEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the BasketballShooterEnv class.')

        self._options = copy.deepcopy(env.unwrapped._options)
        self._options['simulation']['skip_initial_steps'] = 100
        self._options['arm']['upper']['angle'] = -60
        self._options['arm']['upper']['random_angle_offset'] = [0, 0]
        self._options['arm']['lower']['angle'] = 50
        self._options['arm']['lower']['random_angle_offset'] = [0, 0]
        self._options['basket']['random_position_offset']['x'] = [0, 0]
        self._options['basket']['random_position_offset']['y'] = [0, 0]
        self._options['ball']['position_relative'] = True
        self._options['ball']['position'] = (-0.05, 0.1)
        self._options['ball']['random_position_offset']['x'] = [0, 0]
        self._options['ball']['random_position_offset']['y'] = [0, 0]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._action = np.array([0, 0])
        self._action_duration = self._options['simulation']['episode_length']
        if options is not None:
            if 'action' in options:
                self._action = np.array(options['action'])
            if 'duration' in options:
                self._action_duration = options['duration']

        return super().reset(seed=seed, options=self._options)

    def step(self, action: np.ndarray = None):
        if self._env.unwrapped._episode_step < self._action_duration:
            return super().step(self._action)
        else:
            return super().step(np.array([0, 0]))
