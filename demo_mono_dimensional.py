import gym
import gym_mono_dimensional

env = gym.make('MonoDimensional-v0')
env.reset()

for _ in range(100):
    env.step(env.action_space.sample())

