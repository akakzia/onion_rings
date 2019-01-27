import gym
import gym_bi_dimensional
env = gym.make('BiDimensional-v0')

for i_episode in range(10):
    observation = env.reset()
    cum_reward=0
    for t in range(200):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        print(observation,reward,done,info)
        if done:
            print("Episode finished after {} timesteps, final reward : {}".format(t+1,cum_reward))
            break
    if t==199:
        print("Episode not finished after 199 timesteps,final reward : {}".format(cum_reward))

env.close()
