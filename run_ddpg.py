import gym
import numpy as np
import gym_multi_dimensional

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

id = gym_multi_dimensional.dynamic_register(n_dimensions=2, env_description={})
env = gym.make(id)
env = DummyVecEnv([lambda: env])

model = DDPG.load("models/ddpg_n_dimension")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
