import gymnasium as gym
import gymnasium_qr
from gymnasium_qr.wrappers import NoBasket

env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")
env = NoBasket(env)

for episode in range(10):
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

env.close()
