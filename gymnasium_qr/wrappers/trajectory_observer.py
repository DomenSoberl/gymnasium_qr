import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium_qr.envs.basketball_shooter import BasketballShooterEnv
import numpy as np


class TrajectoryObserver(gym.ObservationWrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not BasketballShooterEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the BasketballShooterEnv class.')

        # Add top point (x, y) and vx.
        self._obs_space_size = self.observation_space.shape[0] + 3

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_space_size,),
            dtype=np.float32
        )

    def observation(self, observation: ObsType) -> WrapperObsType:
        (basket_x, basket_y) = self.env.unwrapped._basket.position
        (ball_x, ball_y) = self.env.unwrapped._ball.position

        # After reset
        if self.env.unwrapped.episode_step == 0:
            self._last_x = ball_x
            self._max_x = 0
            self._max_y = 0
            self._max_vx = 0

        # After step
        else:
            trajectory_started = self.env.unwrapped.trajectory_started
            trajectory_ended = self.env.unwrapped.trajectory_ended
            trajectory_active = trajectory_started and not trajectory_ended

            if trajectory_active and ball_y > self._max_y:
                dt = 1 / self.env.unwrapped.metadata['render_fps']
                self._max_x = ball_x
                self._max_y = ball_y
                self._max_vx = (ball_x - self._last_x) / dt

            self._last_x = ball_x

            additional_obs = np.array([self._max_x, self._max_y, self._max_vx], dtype=np.float32)
            return np.concatenate((observation, additional_obs))
