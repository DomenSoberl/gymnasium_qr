from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium_qr.envs.basketball_shooter import BasketballShooterEnv

import copy


class Trajectory(gym.Wrapper):
    def point_at_time(trajectory: list, t: float) -> dict:
        if len(trajectory) < 2:
            return None

        for i in range(len(trajectory) - 1):
            t0 = trajectory[i]["time"]
            t1 = trajectory[i+1]["time"]

            if t0 <= t and t <= t1:
                if (t - t0) < (t1 - t):
                    return trajectory[i]
                else:
                    return trajectory[i+1]

        return None

    def highest_point(trajectory: list) -> dict:
        if len(trajectory) == 0:
            return None

        max_i = 0
        for i in range(len(trajectory) - 1):
            [_, y] = trajectory[i+1]["ball_position"]
            [_, max_y] = trajectory[max_i]["ball_position"]

            if y > max_y:
                max_i = i + 1

        return trajectory[max_i]

    def points_at_height(trajectory: list, y: float) -> (list[dict], list[dict]):
        if len(trajectory) < 2:
            return ([], [])

        points_up = []
        points_down = []
        for i in range(len(trajectory) - 1):
            [x0, y0] = trajectory[i]["ball_position"]
            [x1, y1] = trajectory[i+1]["ball_position"]

            if y0 <= y and y < y1:
                points_up.append(trajectory[i])

            if y0 >= y and y > y1:
                points_down.append(trajectory[i])

        return (points_up, points_down)

    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not BasketballShooterEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the BasketballShooterEnv class.')

        env.unwrapped._pre_render_callbacks.append(self)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        env = self.env.unwrapped

        render_mode = env.render_mode
        env.render_mode = None

        observation, info = super().reset(seed=seed, options=options)

        self._trajectory = [copy.deepcopy(info)]
        info["trajectory"] = self._trajectory

        env.render_mode = render_mode
        if env.render_mode == "human":
            env._render_frame()

        return observation, info

    def step(self, action: np.ndarray = None):
        observation, reward, terminated, truncated, info = super().step(action)

        self._trajectory.append(copy.deepcopy(info))
        info["trajectory"] = self._trajectory

        return observation, reward, terminated, truncated, info

    def _pre_render(self, canvas, ppm):
        env = self.env.unwrapped

        p0 = None
        for info in self._trajectory:
            p1 = info["ball_position"]
            if p0 is not None:
                env._paint_segment(canvas, p0, p1, "yellow", ppm)
            p0 = p1
