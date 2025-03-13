import gymnasium as gym
import gymnasium_qr
from gymnasium_qr.wrappers import Trajectory


env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")
env = Trajectory(env)

for episode in range(10):
    env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

    highest = Trajectory.highest_point(info["trajectory"])
    print(f'Highest point was at {highest["ball_position"]} with velocity {highest["ball_velocity"]}.')

env.close()
