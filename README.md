# Basketball shooter

A Gymnasium-type environment to simulate a two-joint robotic arm throwing a ball to hit a basket.

![BasketballShooter environment](basketball_shooter.png)

The supported rendering modes:

| Mode   | Meaning                              |
|:-------|--------------------------------------|
| human  | Real-time rendering using PyGyme.    |
| png    | Save individual frames as PNG files. |
| None   | Console only, maximum speed.         |

Interaction:

| Space       | Type                                |
|:------------|:------------------------------------|
|Action       | Box(-1.0, 1.0, (2,), float32)       |
|Observation  | Box(-np.inf, np.inf, (4,), float32) |
|Reward       | {0, 1}                              |

The episode length is 300 steps by default. An episode is terminated when the ball hits the basket or moves outside the visible are (left, right or bottom). The upper movement is not restricted. When the ball first starts moving upwards, a trajectory is being drawn until the end of the episode.

### Actions and observations

Actions control the speeds of the two joints (upper and lower). A positive value denotes CCW rotation, a negative value CW rotation.

Observations are the following:

| Index | Quantity                                                                 |
|:-----:|:-------------------------------------------------------------------------|
|0      | Orientation of the upper joint: [-180, 180]                              |
|1      | Orientation of the lower joint, relative to the upper joint: [-180, 180] |
|2      | Position x of the ball in meters: (-inf, inf)                            |
|3      | Position y of the ball in meters: (-inf, inf)                            |

### Reward

Reward is 0 if the basket has not been, 1 otherwise. Reward of 1 can only be obtained in the final step of the episode.

## Usage

Install the Python package with:

`pip install gymnasium_qr@git+https://github.com/DomenSoberl/gymnasium_qr`

A minimum example with random actions:

```
import gymnasium as gym
import gymnasium_qr

env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")

for episode in range(10):
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

env.close()
```

## Custom settings

See the `examples/change_options.py` example for defining the `options` structure. Pass it to the `make` function to permanently set up the environment: 

```
env = gym.make(
    "gymnasium_qr/BasketballShooter-v0",
    render_mode="human",
    options=custom_options
)
```

Or to the `reset` method to override the settings for one episode only:

```
observations, info = env.reset(options=custom_options)
```

## Wrappers

### One shoot

A simplified experimental setup, where only one action of a certain duration is executed at the start of the episode. The action is given with the `reset` method. The `action` parameter given with the `step` function is ignored. Custom settings are partially overridden when given with the `make` method, and ignored if given with the `reset` method.

Example:

```
env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")
env = OneShotScenario(env)

observation, info = env.reset(
    options={
        'action': [0.61, -0.44],
        'duration': 20
    }
)

episode_over = False
while not episode_over:
    observation, reward, terminated, truncated, info = env.step(action=None)
    episode_over = terminated or truncated
```

### Observe trajectory

Extends the observations with two additional variables:

| Index | Quantity                                                           |
|:-----:|:-------------------------------------------------------------------|
|4      | The maximum height reached by the ball: [0, np.inf]                |
|5      | The minimum horizontal distance from the basked: [-np.inf, np.inf] |

The maximum height is being recorded only after the trajectory starts being recorded (from the moment the ball first moves upwards). Before that point it is observed as 0. The minimum horizontal distance from the basket is being observed after the trajectory starts being recorded and up to the moment when the ball passes the vertical alignment with the basket in a downward movement. Before and after it is being locked respectively to the initial and final horizontal offset from the basket. This observation is useful for determining by how much the goal has been missed.

Example:

```
env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")
env = TrajectoryObserver(env)
env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

print(f'Max height: {observation[4]}, goal missed by: {observation[5]}')
```
