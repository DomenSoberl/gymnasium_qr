import gymnasium as gym
import gymnasium_qr
from gymnasium_qr.wrappers import OneShotScenario
from gymnasium_qr.wrappers import TrajectoryObserver


env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")
env = TrajectoryObserver(env)

for episode in range(10):
    env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

    print(f'Max height: {observation[4]}, goal missed by: {observation[5]}')

env.close()
