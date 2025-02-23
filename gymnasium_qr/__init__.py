from gymnasium.envs.registration import register

register(
    id="gymnasium_qr/Basket-v0",
    entry_point="gymnasium_qr.envs:BasketEnv",
)
