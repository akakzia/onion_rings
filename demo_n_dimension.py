import gym
import gym_multi_dimensional

id = gym_multi_dimensional.dynamic_register(n_dimensions=2)
env = gym.make(id)

for i_episode in range(10):
    observation = env.reset()
    cum_reward=0
    episode_nb = 100
    done = False
    for t in range(episode_nb):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        print(observation,reward,done,info)
        if done:
            print("Episode finished after {} timesteps, final reward : {}".format(t+1,cum_reward))
            break
    if not done:
        print("Episode not finished after {} timesteps,final reward : {}".format(episode_nb,cum_reward))

env.close()
