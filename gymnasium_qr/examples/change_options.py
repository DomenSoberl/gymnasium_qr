import gymnasium as gym
from gymnasium_qr.envs import BasketballShooterEnv

default_options = BasketballShooterEnv.get_default_options()
print('Default options:', default_options)

custom_options = {
    'simulation': {
        'world_size': (5, 3),    # The size (width, height) in meters.
        'ppm': 200,              # Pixels per meter for rendering.
        'episode_length': 300,   # Maximum episode length in steps.
        'skip_initial_steps': 0  # Simulate steps before the episode starts. 
    },
    'arm': {
        'position': (1, 1),                   # Arm mount position in meters.
        'upper': {
            'length': 0.5,                    # The length of the upper arm in meters.
            'angle': -60,                     # Initial upper arm angle.
            'random_angle_offset': [-10, 10]  # Random initial offset [min, max] in degrees.
        },
        'lower': {
            'length': 0.5,                    # The length of the lower arm in meters.
            'angle': 30,                      # Initial lower arm angle.
            'random_angle_offset': [-10, 10]  # Random initial offset [min, max] in degrees.
        }
    },
    'basket': {
        'position': (4, 2),          # Position of basket in meters.
        'size': 1.1,                 # The size of the basket relative to the ball.
        'random_position_offset': {  # Random initial offsets [min, max] in meters.
            'x': [0, 0],
            'y': [0, 0]
        }
    },
    'ball': {
        'radius': 0.1,               # The size of the ball (radius) in meters.
        'position': (-0.2, 0.2),     # Initial ball position in meters.
        'position_relative': True,   # Position is relative to the tip od the arm.
        'random_position_offset': {  # Random initial offsets [min, max] in meters.
            'x': [-0.1, 0.1],
            'y': [-0.1, 0.1]
        }
    }
}

env = gym.make(
    "gymnasium_qr/BasketballShooter-v0",
    render_mode="human",
    options=custom_options
)

observations, info = env.reset()

terminated = False
while not terminated:
    observation, reward, terminated, truncated, info = env.step([0, -0.1])
