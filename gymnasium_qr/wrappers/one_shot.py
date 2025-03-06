from typing import Optional
import copy
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium_qr.envs.basketball_shooter import BasketballShooterEnv


class OneShot(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not BasketballShooterEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the BasketballShooterEnv class.')

        self._options = copy.deepcopy(env.unwrapped._options)
        self._options['arm']['position'] = (0.5, 1)
        self._options['arm']['upper']['angle'] = -90
        self._options['arm']['upper']['random_angle_offset'] = [0, 0]
        self._options['arm']['lower']['angle'] = 0
        self._options['arm']['lower']['random_angle_offset'] = [0, 0]
        self._options['basket']['random_position_offset']['x'] = [0, 0]
        self._options['basket']['random_position_offset']['y'] = [0, 0]
        self._options['ball']['position_relative'] = True
        self._options['ball']['position'] = (0.1, 0.1)
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

        env = self.env.unwrapped

        render_mode = env.render_mode
        env.render_mode = None

        observation, info = super().reset(seed=seed, options=self._options)

        (x, y) = env._lower_arm.fixtures[0].shape.vertices[0]
        r = env._ball.fixtures[0].shape.radius

        self._grip = env._lower_arm.CreatePolygonFixture(
            vertices=[
                (x, y),
                (x + 0.02, y),
                (x + 0.02, y + 3*r/2),
                (x, y + 3*r/2)
            ],
            density=1, friction=0.1
        )

        env.render_mode = render_mode
        if env.render_mode == "human":
            env._render_frame()

        return observation, info

    def _release_grip(self):
        if self._grip is not None:
            env = self.env.unwrapped
            env._lower_arm.DestroyFixture(self._grip)
            self._grip = None

    def step(self, action: np.ndarray = None):
        if self.env.unwrapped.episode_step < self._action_duration:
            return super().step(self._action)
        else:
            self._release_grip()
            return super().step(np.array([0, 0]))
