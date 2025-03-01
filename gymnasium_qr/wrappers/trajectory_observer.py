import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium_qr.envs.basketball_shooter import BasketballShooterEnv
import numpy as np


class TrajectoryObserver(gym.ObservationWrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not BasketballShooterEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the BasketballShooterEnv class.')

        # Redefine the observation space to contain:
        # - the highest y point reached,
        # - the final horizontal distance from the goal.
        self._obs_space_size = self.observation_space.shape[0] + 2

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_space_size,),
            dtype=np.float32
        )

    def observation(self, observation: ObsType) -> WrapperObsType:
        # After reset
        if self.env.unwrapped._episode_step == 0:
            if self.env.unwrapped._episode_options is not None:
                options = self.env.unwrapped._episode_options
            else:
                options = self.env.unwrapped._options

            (width, height) = options['simulation']['world_size']
            self._basket_position = options['basket']['position']
            (basket_x, _) = self._basket_position

            self._horz_miss = observation[2] - basket_x
            self._max_y = 0

        # After step
        else:
            x = observation[2]
            y = observation[3]
            (basket_x, basket_y) = self._basket_position

            trajectory_started = self.env.unwrapped._trajectory_started

            if trajectory_started and y > self._max_y:
                self._max_y = y
            if trajectory_started and y >= basket_y:
                self._horz_miss = x - basket_x

            additional_obs = np.array([self._max_y, self._horz_miss], dtype=np.float32)
            return np.concatenate((observation, additional_obs))
