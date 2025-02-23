from gymnasium.envs.registration import register

register(
    id="gymnasium_qr/BasketballShooter-v0",
    entry_point="gymnasium_qr.envs:BasketballShooterEnv",
)
