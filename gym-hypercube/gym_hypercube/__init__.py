from gym.envs.registration import register
from gym_hypercube.envs import hypercube_env as md

def dynamic_register(n_dimensions=2, env_description=md.HypercubeEnv.default_description, continuous=True,
        acceleration=True, reset_radius=None):

    register(id="Hypercube-v0",
             entry_point="gym_hypercube.envs:HypercubeEnv",
             kwargs={"n_dimensions":n_dimensions,
                 "env_description":env_description,
                 "continuous":continuous,
                 "acceleration":acceleration,
                 "reset_radius":reset_radius})

    return "Hypercube-v0"
