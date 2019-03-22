from gym.envs.registration import register
from gym_multi_dimensional.envs import multi_dimensional_env as md

def dynamic_register(n_dimensions=2, env_description=md.MultiDimensionalEnv.default_description, continuous=True,
        acceleration=True, reset_radius=None):

    register(id="MultiDimensional-v0",
             entry_point="gym_multi_dimensional.envs:MultiDimensionalEnv",
             kwargs={"n_dimensions":n_dimensions,
                 "env_description":env_description,
                 "continuous":continuous,
                 "acceleration":acceleration,
                 "reset_radius":reset_radius})

    return "MultiDimensional-v0"
