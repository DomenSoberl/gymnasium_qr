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

The episode length is 300 steps by default. An episode is terminated when the ball hits the basket or falls to the ground.

### Actions and observations

Actions control the speeds of the two joints (upper and lower). A positive value denotes CCW rotation, a negative value CW rotation. The speed can be set within the interval [-10 deg/s, 10 deg/s], which is translated to the interval [-1.0, 1,0] for the input velocity variables.

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

### Trajectory

Draws a trajectory from the moment the ball starts movinf upwards, and until it either touches the basket or the episodes terminates. After the episode finishes, the following methods are available to observe the property of the made trajectory:

The length of the trajectory in steps.  
`trajectory_length() -> int`


The highest point (max y) of the trajectory. If no trajectory has been made, `None` is returned.  
`highest_point(self) -> np.ndarray`


The list of all the points at (or close to) the height y. The first returned list contains the points crossing the horizontal line y in upward motion, the second list in the downward motion.  
`points_at_height(self, y: float) -> (list[np.ndarray], list[np.ndarray])`

The velocity vector at the given point on the trajectory. If the given point is not found on the trajectory, [nan, nan] is returned.  
`velocity_at(self, point: np.ndarray) -> np.ndarray`


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

### No basket

This wrapper removes the basket from the experimental setup. The coordinates are still there and the distance of the ball from the basket is still returned by the 'info' method, but it's physical manifestation is removed from the world.

Usage:

```
from gymnasium_qr.wrappers import NoBasket

env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")
env = NoBasket(env)
```

The rest of the functionality is equivalent to the original class.
