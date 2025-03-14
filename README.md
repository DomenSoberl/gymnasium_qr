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

Reward is 0 if the basket has not been hit, 1 otherwise. Reward of 1 can only be obtained in the final step of the episode.

### Info

The info structure contains the following data:

| key               | Meaning                                            |
|-------------------|----------------------------------------------------|
| `step`            | Episode step (integer)                             |
| `time`            | Episode time in seconds (float)                    |
| `joint_angle`     | Angles of both joints in degrees (array)           |
| `joint_velocity`  | Velocities of both joints in deg/s (array)         |
| `ball_position`   | Position of the ball in meters (array)             | 
| `ball_velocity`   | Velocity of the ball (float)                       |
| `ball_angle`      | Angle of ball's motion (-90 degrees means falling) |
| `basket_position` | The position of the basket in meters (array)       |
| `distance`        | The distance of the ball from the basket (float)   |
| `basket_touched`  | True if the ball touches the basket (collision)    |
| `basket_hit`      | True if the basket is hit (reward = 1)             |

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

Records and draws the trajectory made by the ball. All steps of the episode are recorded. The `info` structure is augmented with the `"trajectory"` key that provides the list of `info` data structures for each consecutive simulation step.

The `Trajectory` class provides three static methods:

`point_at_time(trajectory: list, t: float) -> dict`  
Returns the `info` structure at the time closest to the given `t` (in seconds).

`highest_point(trajectory: list) -> dict`  
Returns the `info` structure of the highest point reached by the ball.

`points_at_height(trajectory: list, y: float) -> (list[dict], list[dict])`  
Returns all the `info` structures of simulation steps that crossed the given height `y`. The first returned list contains the points that crossed ´y´ moving up, and the second list those moving down.

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

This wrapper removes the basket from the experimental setup. The coordinates are still there and the distance of the ball from the basket is still returned by the `info` method, but it's physical manifestation is removed from the world.

Usage:

```
from gymnasium_qr.wrappers import NoBasket

env = gym.make("gymnasium_qr/BasketballShooter-v0", render_mode="human")
env = NoBasket(env)
```

The rest of the functionality is equivalent to the original class.
