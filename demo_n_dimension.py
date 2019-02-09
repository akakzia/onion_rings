import gym
import gym_multi_dimensional
from gym_multi_dimensional.visualization import vis_2d

id = gym_multi_dimensional.dynamic_register(n_dimensions=2,
        env_description={},continuous=True,acceleration=True)
env = gym.make(id)


for i_episode in range(100):
    state = env.reset()
    cum_reward=0
    episode_nb = 200
    done = False
    for t in range(episode_nb):
        #env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        cum_reward += reward
        print(state,reward,done,info)
        if done:
            print("Episode finished after {} timesteps, final reward : {}".format(t+1,cum_reward))
            break
    if not done:
        print("Episode not finished after {} timesteps,final reward : {}".format(episode_nb,cum_reward))


#vis_2d.visualize(rb)

env.close()
