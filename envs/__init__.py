from gym.envs.registration import register
from envs import hypercube_env as md

register(id="Hypercube-v0",
         entry_point="envs.hypercube_env:HypercubeEnv",
         kwargs={"n_dimensions":2,
             "env_description":md.HypercubeEnv.default_description,
             "continuous":True,
             "acceleration":False,
             "reset_radius":0})
