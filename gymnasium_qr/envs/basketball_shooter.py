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
                'episode_length': 300,
                'skip_initial_steps': 0
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
                    'angle': 60,
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
                'weight': 0.5,
                'position': (1.5, 2),
                'position_relative': False,
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
        self._episode_options = None

        # Rendering objects
        self.window = None
        self.clock = None

        # Action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space [upper joint, lower joint, ball x, ball y]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        # Rendering
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.timestep = 1.0/self.metadata['render_fps']

        if render_mode == 'png':
            self._frame_number = 0

        self._pygame_initialized = False
        self.window = None
        self.clock = None
        self._world = None

        self._pre_render_callbacks = []
        self._post_render_callbacks = []

    def _create_world(self):
        # Which options to use?
        if self._episode_options is not None:
            options = self._episode_options
        else:
            options = self._options

        # Create the Box2D world
        world = Box2D.b2.world(gravity=(0, -9.81))

        # Arm
        arm_position = options['arm']['position']
        arm_width = 0.01  # We make it a constant.

        # Upper arm
        arm_length = options['arm']['upper']['length']
        arm_angle = options['arm']['upper']['angle']
        [offset_min, offset_max] = options['arm']['upper']['random_angle_offset']

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
            density=50, friction=1
        )

        # Lower arm
        arm_length = options['arm']['lower']['length']
        arm_angle = options['arm']['upper']['angle'] + options['arm']['lower']['angle']
        [offset_min, offset_max] = options['arm']['lower']['random_angle_offset']

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
            density=50, friction=1
        )

        # Joint between the mount and the upper arm
        joint1 = world.CreateRevoluteJoint(
            bodyA=arm_mount,
            bodyB=upper_arm,
            localAnchorA=(0, 0),
            localAnchorB=(0, 0),
            enableMotor=True,
            motorSpeed=0,
            maxMotorTorque=100
        )

        # Joint between the upper and the lower arm
        joint2 = world.CreateRevoluteJoint(
            bodyA=upper_arm,
            bodyB=lower_arm,
            localAnchorA=(options['arm']['upper']['length'], 0),
            localAnchorB=(0, 0),
            enableMotor=True,
            motorSpeed=0,
            maxMotorTorque=100
        )

        # Basket
        (position_x, position_y) = options['basket']['position']
        [offset_x_min, offset_x_max] = options['basket']['random_position_offset']['x']
        [offset_y_min, offset_y_max] = options['basket']['random_position_offset']['y']
        r = options['basket']['size'] * options['ball']['radius']
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
        (position_x, position_y) = options['ball']['position']
        if options['ball']['position_relative']:
            (origin_x, origin_y) = lower_arm.transform * (options['arm']['lower']['length'], 0)
            position_x += origin_x
            position_y += origin_y

        [offset_x_min, offset_x_max] = options['ball']['random_position_offset']['x']
        [offset_y_min, offset_y_max] = options['ball']['random_position_offset']['y']

        ball_density = (options['ball']['weight']/(math.pi * options['ball']['radius']**2))

        ball = world.CreateDynamicBody(
            position=(
                position_x + np.random.uniform(offset_x_min, offset_x_max),
                position_y + np.random.uniform(offset_y_min, offset_y_max)
            )
        )
        ball.CreateCircleFixture(
            radius=options['ball']['radius'],
            density=ball_density,
            friction=1
        )

        self._world = world
        self._upper_arm = upper_arm
        self._lower_arm = lower_arm
        self._joint1 = joint1
        self._joint2 = joint2
        self._basket = basket
        self._ball = ball

    def _get_obs(self):
        joint1_angle = math.degrees(self._upper_arm.angle)
        joint2_angle = math.degrees(self._lower_arm.angle) - joint1_angle
        ball_x = self._ball.position.x
        ball_y = self._ball.position.y

        return np.array([joint1_angle, joint2_angle, ball_x, ball_y], dtype=np.float32)

    def _get_info(self):
        joint1_angle = math.degrees(self._upper_arm.angle) % 360 - 180
        joint2_angle = joint1_angle + math.degrees(self._lower_arm.angle) % 360 - 180

        ball_x = self._ball.position.x
        ball_y = self._ball.position.y
        basket_x = self._basket.position.x
        basket_y = self._basket.position.y

        distance = math.dist((ball_x, ball_y), (basket_x, basket_y))

        basket_collision = False
        if len(self._basket.contacts) > 0:
            for contact in self._basket.contacts:
                for point in contact.contact.worldManifold.points:
                    if point != (0, 0):
                        basket_collision = True

        return {
            "step": self.episode_step,
            "joint_angle": np.array([joint1_angle, joint2_angle], dtype=np.float32),
            "joint_velocity": np.array([self._joint1_vel, self._joint2_vel], dtype=np.float32),
            "ball_position": np.array([ball_x, ball_y], dtype=np.float32),
            "ball_velocity": self._ball_vel,
            "ball_angle": self._ball_ang,
            "basket_position": np.array([basket_x, basket_y], dtype=np.float32),
            "distance": distance,
            "basket_touched": basket_collision,
            "basket_hit": (distance < 0.01)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if options is not None:
            self._episode_options = options
        else:
            self._episode_options = None
            options = self._options

        if self._world is not None:
            for body in self._world.bodies:
                self._world.DestroyBody(body)
            self._world = None

        self._create_world()

        skip_steps = options['simulation']['skip_initial_steps']

        for _ in range(skip_steps):
            self._world.Step(
                timeStep=self.timestep,
                velocityIterations=4, positionIterations=4
            )

        self.episode_step = 0
        self.last_observation = None

        self._joint1_vel = 0
        self._joint2_vel = 0
        self._ball_vel_x = 0
        self._ball_vel_y = 0
        self._ball_vel = 0
        self._ball_ang = 0

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
            timeStep=self.timestep,
            velocityIterations=20, positionIterations=20
        )

        self.episode_step += 1

        observation = self._get_obs()
        info = self._get_info()

        basket_collision = info['basket_touched']

        if self.last_observation is not None:
            [angle1_0, angle2_0, x0, y0] = self.last_observation
            [angle1_1, angle2_1, x1, y1] = self._get_obs()

            joint1_vel = angle1_1 - angle1_0
            joint2_vel = angle2_1 - angle2_0

            if joint1_vel > 180:
                joint1_vel -= 360
            if joint1_vel < -180:
                joint1_vel += 360
            if joint2_vel > 180:
                joint2_vel -= 360
            if joint2_vel < -180:
                joint2_vel += 360

            self._joint1_vel = joint1_vel / self.timestep
            self._joint2_vel = joint2_vel / self.timestep

            ball_dx = x1 - x0
            ball_dy = y1 - y0
            self._ball_vel_x = ball_dx / self.timestep
            self._ball_vel_y = ball_dy / self.timestep
            self._ball_vel = math.dist((x0, y0), (x1, y1)) / self.timestep
            self._ball_ang = math.degrees(math.atan2(ball_dy, ball_dx))

        if self.render_mode == "human" or self.render_mode == "png":
            self._render_frame()

        self.last_observation = observation

        reward = 1 if info['distance'] < 0.01 else 0

        [ball_x, ball_y] = info['ball_position']
        (width, height) = self._options['simulation']['world_size']

        terminated = bool(
            info['distance'] < 0.01 or ball_y < 0
        )

        truncated = bool(
            self.episode_step >= self._options['simulation']['episode_length']
        )

        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    def _render_frame(self):
        # Which options to use?
        if self._episode_options is not None:
            options = self._episode_options
        else:
            options = self._options

        (width, height) = options['simulation']['world_size']
        ppm = options['simulation']['ppm']

        if not self._pygame_initialized:
            pygame.init()
            self._pygame_initialized = True

        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (round(width * ppm), round(height * ppm))
            )
            pygame.display.set_caption('Basketball shooter')

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((round(width * ppm), round(height * ppm)))
        canvas.fill((0, 0, 0))

        for callback in self._pre_render_callbacks:
            callback._pre_render(canvas, ppm)

        self._paint_body(canvas, self._upper_arm, "blue", ppm)
        self._paint_body(canvas, self._lower_arm, "blue", ppm)
        self._paint_circle(canvas, self._upper_arm.position, 0.2, "dark blue", ppm)
        self._paint_circle(canvas, self._lower_arm.position, 0.2, "dark blue", ppm)
        self._paint_body(canvas, self._basket, "dark green", ppm)
        self._paint_body(canvas, self._ball, "dark red", ppm)

        for callback in self._post_render_callbacks:
            callback._post_render(canvas, ppm)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == 'png':
            pygame.image.save(canvas, f'frame{self._frame_number}.png')
            self._frame_number += 1

    def _paint_circle(self, canvas, position, radius, color, ppm):
        (x, y) = position
        vx = round(x * ppm)
        vy = canvas.get_height() - round(y * ppm)
        pygame.draw.circle(canvas, color, (vx, vy), 10)

    def _paint_segment(self, canvas, p0, p1, color, ppm):
        (x0, y0) = p0
        vx0 = round(x0 * ppm)
        vy0 = canvas.get_height() - round(y0 * ppm)

        (x1, y1) = p1
        vx1 = round(x1 * ppm)
        vy1 = canvas.get_height() - round(y1 * ppm)

        pygame.draw.line(canvas, color, (vx0, vy0), (vx1, vy1), 1)

    def _paint_body(self, canvas, body, color, ppm):
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
