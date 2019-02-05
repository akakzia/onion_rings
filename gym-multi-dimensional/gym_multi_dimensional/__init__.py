from gym.envs.registration import register

def dynamic_register(n_dimensions=1, env_description={},continuous=True,
        acceleration=True):
    register(id="MultiDimensional-v{}".format(n_dimensions),
             entry_point="gym_multi_dimensional.envs:MultiDimensionalEnv",
             kwargs={"n_dimensions":n_dimensions,
                 "env_description":env_description,
                 "continuous":continuous,
                 "acceleration":acceleration})
    return "MultiDimensional-v{}".format(n_dimensions)
