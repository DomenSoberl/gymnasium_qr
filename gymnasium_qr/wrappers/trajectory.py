import math
from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium_qr.envs.basketball_shooter import BasketballShooterEnv


class Trajectory(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not BasketballShooterEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the BasketballShooterEnv class.')

        env.unwrapped._pre_render_callbacks.append(self)

    def trajectory_length(self) -> int:
        return len(self._trajectory)

    def highest_point(self) -> np.ndarray:
        if len(self._trajectory) == 0:
            return None

        max_p = None
        for (p, _) in self._trajectory:
            if max_p is None:
                max_p = p
                continue

            (_, y) = p
            (_, max_y) = max_p

            if y > max_y:
                max_p = p

        return max_p

    def points_at_height(self, y: float) -> (list[np.ndarray], list[np.ndarray]):
        if len(self._trajectory) < 2:
            return ([], [])

        points_up = []
        points_down = []
        for i in range(len(self._trajectory) - 1):
            (p0, _) = self._trajectory[i]
            (p1, _) = self._trajectory[i+1]

            [_, y0] = p0
            [_, y1] = p1

            if y0 <= y and y < y1:
                points_up.append(p0)

            if y0 >= y and y > y1:
                points_down.append(p0)

        return (points_up, points_down)

    def velocity_at(self, point: np.ndarray) -> np.ndarray:
        if point is None or len(self._trajectory) < 2:
            return np.array([np.nan, np.nan])

        [x, y] = point

        for i in range(len(self._trajectory) - 1):
            (p0, _) = self._trajectory[i]
            (p1, dt) = self._trajectory[i+1]

            [x0, y0] = p0
            [x1, y1] = p1

            if x0 == x and y0 == y:
                return np.array([(x1 - x0)/dt, (y1 - y0)/dt])

        return np.array([np.nan, np.nan])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        env = self.env.unwrapped

        self._trajectory_started = False
        self._trajectory_ended = False
        self._trajectory = []

        render_mode = env.render_mode
        env.render_mode = None

        observation, info = super().reset(seed=seed, options=options)

        env.render_mode = render_mode
        if env.render_mode == "human":
            env._render_frame()

        return observation, info

    def step(self, action: np.ndarray = None):
        env = self.env.unwrapped

        observation, reward, terminated, truncated, info = super().step(action)

        if not self._trajectory_started and env._ball_vel_y > 0:
            self._trajectory_started = True

        if self._trajectory_started and not self._trajectory_ended:
            self._trajectory.append((info["ball_position"], env.timestep))

        if not self._trajectory_ended and info["basket_touched"]:
            self._trajectory_ended = True

        return observation, reward, terminated, truncated, info

    def _pre_render(self, canvas, ppm):
        env = self.env.unwrapped

        p0 = None
        for (p1, _) in self._trajectory:
            if p0 is not None:
                env._paint_segment(canvas, p0, p1, "yellow", ppm)
            p0 = p1
