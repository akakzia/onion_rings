from gym.envs.registration import register

register(
        id="MultiDimensional-v0",
        entry_point="gym_multi_dimensional.envs:MultiDimensionalEnv"
        )
