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

    interval = env.trajectory_time_interval()
    if interval is not None:
        (t0, t1) = interval
        print(f'Trajectory was recorded between {t0} and {t1} seconds.')

        vel = env.velocity_at(env.point_at_time(t1))
        print(f'Final velocity was {vel}.')

    p = env.highest_point()
    vel = env.velocity_at(p)
    print(f'Highest point {p} with velocity {vel}.')

    (points_up, points_down) = env.points_at_height(1)

    print(f'There are {len(points_up)} points at height 1 m moving up.')
    for p in points_up:
        print(f'{p} with velocity {env.velocity_at(p)} at angle {env.angle_at(p)}.')

    print(f'There are {len(points_down)} points at height 1 m moving down.')
    for p in points_down:
        print(f'{p} with velocity {env.velocity_at(p)} at angle {env.angle_at(p)}.')

env.close()
