import math
from typing import Optional

import gymnasium as gym
import pygame
import Box2D
import numpy as np


class BasketballShooterEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "png"],
        "render_fps": 60,
    }

    def get_default_options():
        return {
            'simulation': {
                'world_size': (5, 3),
                'ppm': 200,
                'episode_length': 300
            },
            'arm': {
                'position': (1, 1),
                'upper': {
                    'length': 0.5,
                    'angle': -60,
                    'random_angle_offset': [-10, 10]
                },
                'lower': {
                    'length': 0.5,
                    'angle': 30,
                    'random_angle_offset': [-10, 10]
                }
            },
            'basket': {
                'position': (4, 2),
                'size': 1.1,
                'random_position_offset': {
                    'x': [0, 0],
                    'y': [0, 0]
                }
            },
            'ball': {
                'radius': 0.1,
                'position': (1.5, 1),
                'random_position_offset': {
                    'x': [-0.1, 0.1],
                    'y': [-0.1, 0.1]
                }
            }
        }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        options: Optional[dict] = None
    ):
        self._options = (
            options if options is not None
            else BasketballShooterEnv.get_default_options()
        )

        # Rendering objects
        self.window = None
        self.clock = None

        # Action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space [upper joint, lower joint, ball x, ball y]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        if render_mode == 'png':
            self._frame_number = 0

        self._pygame_initialized = False
        self.window = None
        self.clock = None

        self._world = None

    def _create_world(self):
        # Create the Box2D world
        world = Box2D.b2.world(gravity=(0, -9.81))

        # Arm
        arm_position = self._options['arm']['position']
        arm_width = 0.01  # We make it a constant.

        # Upper arm
        arm_length = self._options['arm']['upper']['length']
        arm_angle = self._options['arm']['upper']['angle']
        [offset_min, offset_max] = self._options['arm']['upper']['random_angle_offset']

        arm_mount = world.CreateStaticBody(position=arm_position)
        upper_arm = world.CreateDynamicBody(
            position=arm_position,
            angle=math.radians(arm_angle + np.random.uniform(offset_min, offset_max))
        )
        upper_arm.CreatePolygonFixture(
            vertices=[
                (0, -arm_width),
                (arm_length, -arm_width),
                (arm_length, arm_width),
                (0, arm_width)
            ],
            density=1, friction=0.1
        )

        # Lower arm
        arm_length = self._options['arm']['lower']['length']
        arm_angle = self._options['arm']['lower']['angle']
        [offset_min, offset_max] = self._options['arm']['lower']['random_angle_offset']

        lower_arm = world.CreateDynamicBody(
            position=(upper_arm.transform * (arm_length, 0)),
            angle=math.radians(arm_angle + np.random.uniform(offset_min, offset_max))
        )
        lower_arm.CreatePolygonFixture(
            vertices=[
                (0, -arm_width),
                (arm_length, -arm_width),
                (arm_length, arm_width),
                (0, arm_width)
            ],
            density=1, friction=0.1
        )
        lower_arm.CreatePolygonFixture(
            vertices=[
                (arm_length, -arm_width),
                (1.01 * arm_length, -arm_width),
                (1.01 * arm_length, 2 * arm_width),
                (arm_length, 2 * arm_width)
            ],
            density=1, friction=0.1
        )

        # Joint between the mount and the upper arm
        joint1 = world.CreateRevoluteJoint(
            bodyA=arm_mount,
            bodyB=upper_arm,
            localAnchorA=(0, 0),
            localAnchorB=(0, 0),
            enableMotor=True,
            motorSpeed=0,
            maxMotorTorque=10000
        )

        # Joint between the upper and the lower arm
        joint2 = world.CreateRevoluteJoint(
            bodyA=upper_arm,
            bodyB=lower_arm,
            localAnchorA=(self._options['arm']['upper']['length'], 0),
            localAnchorB=(0, 0),
            enableMotor=True,
            motorSpeed=0,
            maxMotorTorque=10000
        )

        # Basket
        (position_x, position_y) = self._options['basket']['position']
        [offset_x_min, offset_x_max] = self._options['basket']['random_position_offset']['x']
        [offset_y_min, offset_y_max] = self._options['basket']['random_position_offset']['y']
        r = self._options['basket']['size'] * self._options['ball']['radius']
        w = 0.02  # We make the width of the basked line aconstant.

        basket = world.CreateStaticBody(
            position=(
                position_x + np.random.uniform(offset_x_min, offset_x_max),
                position_y + np.random.uniform(offset_y_min, offset_y_max)
            ),
            shapes=[
                Box2D.b2.polygonShape(vertices=[(-r, +r), (-r-w, +r), (-r-w, -r-w), (-r, -r-w)]),
                Box2D.b2.polygonShape(vertices=[(-r, -r), (-r, -r-w), (+r, -r-w), (+r, -r)]),
                Box2D.b2.polygonShape(vertices=[(+r, -r-w), (+r+w, -r-w), (+r+w, +r), (+r, +r)])
            ]
        )

        # Ball
        (position_x, position_y) = self._options['ball']['position']
        [offset_x_min, offset_x_max] = self._options['ball']['random_position_offset']['x']
        [offset_y_min, offset_y_max] = self._options['ball']['random_position_offset']['y']

        ball = world.CreateDynamicBody(
            position=(
                position_x + np.random.uniform(offset_x_min, offset_x_max),
                position_y + np.random.uniform(offset_y_min, offset_y_max)
            )
        )
        ball.CreateCircleFixture(
            radius=self._options['ball']['radius'],
            density=0.01, friction=0.1
        )

        self._world = world
        self._upper_arm = upper_arm
        self._lower_arm = lower_arm
        self._joint1 = joint1
        self._joint2 = joint2
        self._basket = basket
        self._ball = ball

    def _get_obs(self):
        joint1_angle = (math.degrees(self._upper_arm.angle) % 360) / 360
        joint2_angle = (math.degrees(self._lower_arm.angle) % 360) / 360

        (width, height) = self._options['simulation']['world_size']
        ball_x = self._ball.position.x / width
        ball_y = self._ball.position.y / height

        return np.array([joint1_angle, joint2_angle, ball_x, ball_y], dtype=np.float32)

    def _get_info(self):
        joint1_angle = math.degrees(self._upper_arm.angle) % 360
        joint2_angle = math.degrees(self._lower_arm.angle) % 360

        ball_x = self._ball.position.x
        ball_y = self._ball.position.y
        goal_x = self._basket.position.x
        goal_y = self._basket.position.y

        return {
            "step": self._episode_step,
            "joints": np.array([joint1_angle, joint2_angle], dtype=np.float32),
            "ball": np.array([ball_x, ball_y], dtype=np.float32),
            "goal": np.array([goal_x, goal_y], dtype=np.float32),
            "distance": math.dist((ball_x, ball_y), (goal_x, goal_y))
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self._world is not None:
            for body in self._world.bodies:
                self._world.DestroyBody(body)
            self._world = None

        self._create_world()

        skip_steps = 0
        if options is not None and 'skip_steps' in options:
            skip_steps = options['skip_steps']

        for _ in range(skip_steps):
            self._world.Step(
                timeStep=1.0/self.metadata['render_fps'],
                velocityIterations=4, positionIterations=4
            )

        self._episode_step = 0
        self._trajectory = []

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: np.ndarray):
        if self._world is None:
            return

        np.clip(action, -1, 1)

        self._joint1.motorSpeed = float(action[0] * 10)
        self._joint2.motorSpeed = float(action[1] * 10)

        self._world.Step(
            timeStep=1.0/self.metadata['render_fps'],
            velocityIterations=4, positionIterations=4
        )

        self._episode_step += 1
        self._trajectory.append((self._ball.position.x, self._ball.position.y))

        if self.render_mode == "human" or self.render_mode == "png":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()

        reward = 1 if info['distance'] < 0.01 else 0

        [ball_x, ball_y] = info['ball']
        (width, height) = self._options['simulation']['world_size']

        terminated = bool(
            info['distance'] < 0.01 or
            ball_x < 0 or ball_x > width or ball_y < 0
        )

        truncated = bool(
            self._episode_step >= self._options['simulation']['episode_length']
        )

        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    def _render_frame(self):
        (width, height) = self._options['simulation']['world_size']
        ppm = self._options['simulation']['ppm']

        if not self._pygame_initialized:
            pygame.init()
            self._pygame_initialized = True

        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (round(width * ppm), round(height * ppm))
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((round(width * ppm), round(height * ppm)))
        canvas.fill((0, 0, 0))

        self._paint_body(canvas, self._upper_arm, "blue")
        self._paint_body(canvas, self._lower_arm, "blue")
        self._paint_circle(canvas, self._upper_arm.position, 0.2, "dark blue")
        self._paint_circle(canvas, self._lower_arm.position, 0.2, "dark blue")
        self._paint_body(canvas, self._basket, "dark green")
        self._paint_body(canvas, self._ball, "dark red")

        p0 = None
        for p1 in self._trajectory:
            if p0 is not None:
                self._paint_segment(canvas, p0, p1, "yellow")
            p0 = p1

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == 'png':
            pygame.image.save(canvas, f'frame{self._frame_number}.png')
            self._frame_number += 1

    def _paint_circle(self, canvas, position, radius, color):
        ppm = self._options['simulation']['ppm']

        (x, y) = position
        vx = round(x * ppm)
        vy = canvas.get_height() - round(y * ppm)
        pygame.draw.circle(canvas, color, (vx, vy), 10)

    def _paint_segment(self, canvas, p0, p1, color):
        ppm = self._options['simulation']['ppm']

        (x0, y0) = p0
        vx0 = round(x0 * ppm)
        vy0 = canvas.get_height() - round(y0 * ppm)

        (x1, y1) = p1
        vx1 = round(x1 * ppm)
        vy1 = canvas.get_height() - round(y1 * ppm)

        pygame.draw.line(canvas, color, (vx0, vy0), (vx1, vy1), 1)

    def _paint_body(self, canvas, body, color):
        ppm = self._options['simulation']['ppm']
        height = canvas.get_height()

        for fixture in body.fixtures:
            if fixture.shape is None:
                continue

            if isinstance(fixture.shape, Box2D.b2CircleShape):
                x = round(body.position.x * ppm)
                y = height - round(body.position.y * ppm)
                r = fixture.shape.radius * ppm
                pygame.draw.circle(canvas, color, (x, y), r)
            elif isinstance(fixture.shape, Box2D.b2PolygonShape):
                transformed_shape = [body.transform * v for v in fixture.shape.vertices]
                shape = [
                    (round(vx * ppm), height - round(vy * ppm))
                    for (vx, vy) in transformed_shape
                ]
                pygame.draw.polygon(canvas, color, shape)
            else:
                pass  # Unknow shape

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
