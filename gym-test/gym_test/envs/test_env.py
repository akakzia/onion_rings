import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class TestEnv(gym.Env):
    """
    Description:
        An agent is attached by an un-actuated joint to a cart, which moves
        along a track. The goal is to find the flag by increasing and reducing
        the cart's velocity.

    Observation: 
        Type: Box(4)
        Num Observation     Min     Max
        0   Agent Position  -1      1
        1   Agent Velocity  -0.1    0.1

    Actions:
        Type: Discrete(3)
        Num Action
        0   Push agent to the left
        1   Don't Push agent
        2   Push agent to the right
        
    Reward:
        Reward is 1 if the agent find the green flag
        Raward is 0.1 if the agent find the red flag

    Starting State
        Agent position is assigned a uniform random value between -0.6 and-0.4
        Agent velocity is assigned to 0

    Episode Termination:
        Agent find any flag
        Episode length is greater than 100
    """

    metadata = {
            'render.modes': ['human','rgb_array'],
            'video.frames_per_second': 30
            }


    def __init__(self):
        self.max_position = 1
        self.max_speed = 0.1
        
        self.high = np.array([self.max_position, self.max_speed])
        self.low = -self.high

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
                self.low, self.high, dtype=np.float32)

        self.viewer = None
        self.state = None
        
        self.seed()
        self.reset()


    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
                action, type(action))
        position, velocity = self.state
        velocity += (action-1)*0.01
        if velocity>0:
            velocity = max(0,velocity-0.001)
        else:
            velocity = min(0,velocity+0.001)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, -self.max_position, self.max_position)

        green_flag = bool(position >= self.max_position)
        red_flag = bool(position <= -self.max_position)
        done = bool(green_flag or red_flag)
        if green_flag:
            reward = 1
            info = "green flag"
        elif red_flag:
            reward =  0.1
            info = "red flag"
        else:
            reward = -0.01
            info = ""

        self.state = (position,velocity)
        return np.array(self.state), reward, done, info


    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 200

        world_width = self.max_position*2
        scale = screen_width/world_width

        agenty = 100
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #track
            tracky = 100
            self.track = rendering.Line((0,tracky),(screen_width,tracky))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            #agent
            agentwidth = 20
            agentheight = 20
            agenty = tracky
            l,r,t,b = (-agentwidth/2, agentwidth/2, agentheight/2,
                    -agentheight/2)
            self.agent = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
            self.agenttrans = rendering.Transform()
            self.agent.add_attr(self.agenttrans)
            self.viewer.add_geom(self.agent)
            
            #greenflag
            greenflagx1 = screen_width-10
            greenflagx2 = screen_width
            greenflagy1 = tracky+20
            greenflagy2 = tracky-20
            self.greenflag = rendering.FilledPolygon(
                    [(greenflagx1, greenflagy1),
                        (greenflagx2, greenflagy1),
                        (greenflagx2, greenflagy2),
                        (greenflagx1, greenflagy2)]
                    )
            self.greenflag.set_color(0,1,0)
            self.viewer.add_geom(self.greenflag)
            
            #redflag
            redflagx1 = 0
            redflagx2 = 10
            redflagy1 = tracky+20
            redflagy2 = tracky-20
            self.redflag = rendering.FilledPolygon(
                    [(redflagx1, redflagy1),
                        (redflagx2, redflagy1),
                        (redflagx2, redflagy2),
                        (redflagx1, redflagy2)]
                    )
            self.redflag.set_color(1,0,0)
            self.viewer.add_geom(self.redflag)


        if self.state is None: return None

        pos = self.state[0]
        agentx = (pos+self.max_position)*scale
        self.agenttrans.set_translation(agentx,agenty)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
