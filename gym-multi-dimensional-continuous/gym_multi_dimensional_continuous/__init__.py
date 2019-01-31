from gym.envs.registration import register

def dynamic_register(n_dimensions=1, env_description={}):
    register(id="MultiDimensionalContinuous-{}-v0".format(n_dimensions),
             entry_point="gym_multi_dimensional_continuous.envs:MultiDimensionalContinuousEnv",
             kwargs={"n_dimensions":n_dimensions, "env_description":env_description})
    return "MultiDimensionalContinuous-{}-v0".format(n_dimensions)
