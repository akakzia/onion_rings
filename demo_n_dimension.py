import gym
import gym_multi_dimensional
import replay_buffer

rb = replay_buffer.ReplayBuffer(1000)

id = gym_multi_dimensional.dynamic_register(n_dimensions=2,
        env_description={},continuous=True,acceleration=True)
env = gym.make(id)

old_observation = None

for i_episode in range(20):
    old_observation = env.reset()
    cum_reward=0
    episode_nb = 200
    done = False
    for t in range(episode_nb):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rb.push(old_observation, action, reward, observation)
        old_observation = observation
        cum_reward += reward
        print(observation,reward,done,info)
        if done:
            print("Episode finished after {} timesteps, final reward : {}".format(t+1,cum_reward))
            break
    if not done:
        print("Episode not finished after {} timesteps,final reward : {}".format(episode_nb,cum_reward))

env.close()
