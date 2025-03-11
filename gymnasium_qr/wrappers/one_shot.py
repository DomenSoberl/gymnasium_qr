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

    def _append_info(self, info):
        info["release_time"] = self._release_time

        if self._release_data is not None:
            (position, velocity, angle) = self._release_data
            info["relesed"] = True
            info["release_ball_position"] = position
            info["release_ball_velocity"] = velocity
            info["release_ball_angle"] = angle
        else:
            info["relesed"] = False
            info["release_ball_position"] = np.array([0, 0], dtype=np.float32)
            info["release_ball_velocity"] = 0
            info["release_ball_angle"] = 0

        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._action = np.array([0, 0])
        self._action_duration = self._options['simulation']['episode_length']
        if options is not None:
            if 'action' in options:
                self._action = np.array(options['action'])
            if 'duration' in options:
                self._release_time = round(options['duration'], 3)

        env = self.env.unwrapped
        self._release_data = None

        render_mode = env.render_mode
        env.render_mode = None

        observation, info = super().reset(seed=seed, options=self._options)
        info = self._append_info(info)

        (x, y) = env._lower_arm.fixtures[0].shape.vertices[0]
        r = env._ball.fixtures[0].shape.radius

        grip1 = env._lower_arm.CreatePolygonFixture(
            vertices=[
                (x, y),
                (x + 0.02, y),
                (x + 0.02, y + 2*r + 0.02),
                (x, y + 2*r + 0.02)
            ],
            density=1, friction=1
        )

        grip2 = env._lower_arm.CreatePolygonFixture(
            vertices=[
                (x + 0.02, y + 2*r + 0.02),
                (x + 0.02, y + 2*r + 0.04),
                (x + 0.02 - r, y + 2*r + 0.04),
                (x + 0.02 - r, y + 2*r + 0.02)
            ],
            density=1, friction=1
        )

        self._grip = (grip1, grip2)

        env.render_mode = render_mode
        if env.render_mode == "human":
            env._render_frame()

        self._current_time = 0

        return observation, info

    def _release_grip(self):
        if self._grip is not None:
            env = self.env.unwrapped
            (grip1, grip2) = self._grip
            env._lower_arm.DestroyFixture(grip1)
            env._lower_arm.DestroyFixture(grip2)
            self._grip = None

    def _store_release_data(self):
        info = self.env.unwrapped._get_info()
        (ball_x, ball_y) = info["ball_position"]
        self._release_data = (
            np.array([ball_x, ball_y], dtype=np.float32),
            float(info["ball_velocity"]),
            float(info["ball_angle"])
        )

    def step(self, action: np.ndarray = None):
        env = self.env.unwrapped

        t0 = self._current_time                 # Current time.
        t1 = self._current_time + env.timestep  # Time at next step.
        tr = self._release_time                 # Release time.

        if t0 <= tr and tr < t1:
            render_mode = env.render_mode
            env.render_mode = None

            episode_step = env.episode_step
            timestep = env.timestep

            step1 = tr - t0
            step2 = t1 - tr

            if step1 > 0:
                env.timestep = step1
                observation, reward, terminated, truncated, info = super().step(self._action)
                info = self._append_info(info)

            self._release_grip()
            self._store_release_data()

            if step2 > 0:
                env.timestep = step2
                observation, reward, terminated, truncated, info = super().step([0, 0])
                info = self._append_info(info)

            env.timestep = timestep

            env.render_mode = render_mode
            if env.render_mode == "human":
                env._render_frame()

            env.episode_step = episode_step + 1

        elif t0 < tr:
            observation, reward, terminated, truncated, info = super().step(self._action)
            info = self._append_info(info)

        else:  # tr < t0
            observation, reward, terminated, truncated, info = super().step(np.array([0, 0]))
            info = self._append_info(info)

        self._current_time = t1

        return observation, reward, terminated, truncated, info
