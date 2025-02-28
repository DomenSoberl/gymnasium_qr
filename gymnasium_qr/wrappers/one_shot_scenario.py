import gymnasium as gym
from gymnasium_qr.envs.basketball_shooter import BasketballShooterEnv
import numpy as np
import copy


class OneShotScenario(gym.Wrapper):
    def __init__(self, env: BasketballShooterEnv):
        super().__init__(env)
        self._env = env

        self._options = copy.deepcopy(env.unwrapped._options)
        self._options['simulation']['skip_initial_steps'] = 100
        self._options['arm']['upper']['angle'] = -60
        self._options['arm']['upper']['random_angle_offset'] = [0, 0]
        self._options['arm']['lower']['angle'] = -10
        self._options['arm']['lower']['random_angle_offset'] = [0, 0]
        self._options['basket']['random_position_offset']['x'] = [0, 0]
        self._options['basket']['random_position_offset']['y'] = [0, 0]
        self._options['ball']['position_relative'] = True
        self._options['ball']['position'] = (-0.05, 0.1)
        self._options['ball']['random_position_offset']['x'] = [0, 0]
        self._options['ball']['random_position_offset']['y'] = [0, 0]

    def reset(self):
        self._action = np.array([0, 0])
        return super().reset(options=self._options)

    def set_action(self, action: np.ndarray, duration: int = None):
        self._action = action
        if duration is not None:
            self._action_duration = duration
        else:
            self._action_duration = self._options['simulation']['episode_length']

    def step(self):
        if self._env.unwrapped._episode_step < self._action_duration:
            return super().step(self._action)
        else:
            return super().step(np.array([0, 0]))
