import gymnasium as gym
import gymnasium_qr
from gymnasium_qr.wrappers import OneShot


env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")
env = OneShot(env)

observation, info = env.reset(
    options={
        'action': [0.45, 0.65],
        'duration': 0.13
    }
)

episode_over = False
while not episode_over:
    observation, reward, terminated, truncated, info = env.step(action=None)
    episode_over = terminated or truncated

env.close()
