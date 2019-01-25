import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding

class MonoDimensional(gym.Env):
    metadata = {'render.modes': ['human']}
    position = 0
    step_count = 0
    last_reward = 0
    last_observation = None
    done = False

    def __init__(self):
        self.reset()

    def step(self, action):

        self.position += action
        self.step_count += 1
        self.compute_observation();
        self.compute_reward();
        info = self.get_info();

        return self.last_observation, self.last_reward, self.done, info

    def compute_observation(self):
        return None

    def compute_reward(self):

        """ Left edge reached : low reward """
        if self.position <= 0:
            self.done = True
            self.last_reward = 1

        if self.position >= 1:
            self.done = True
            self.last_reward = 2

        self.last_reward = 0

    def get_info(self):
        return {'Step count': self.step_count, 'Position': self.position, 'Reward':self.last_reward}

    def reset(self):
        self.position = random.uniform(0, 1)
        self.done = False
        self.step_count = 0
        self.last_reward = 0

    def render(self, modes='human', close=False):
        pass

    @property
    def action_space(self):
        class ActionSpace:
            def sample(actionSpace_self, max_magnitude=0.05):
                return random.uniform(0, max_magnitude)
        return ActionSpace()


