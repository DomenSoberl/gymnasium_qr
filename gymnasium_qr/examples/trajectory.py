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

    trajectory = info["trajectory"]

    highest = Trajectory.highest_point(trajectory)
    print(f'Highest point was at {highest["ball_position"]} with velocity {highest["ball_velocity"]}.')

    [basket_x, basket_y] = info['basket_position']
    dist = Trajectory.distance_to_point(trajectory, (basket_x, basket_y))

    if dist is not None:
        print(f'The distance from the basket was {dist}.')

env.close()
