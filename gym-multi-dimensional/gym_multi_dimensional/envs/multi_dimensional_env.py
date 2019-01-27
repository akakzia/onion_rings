import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class MultiDimensionalEnv(gym.Env):
    """
    Description:
        An agent is in a n-cube (n dimension hypercube). The goal is to reach
        the green (n-1)-face of the n-cube (the green vertex of the segment if
        n=1, the green edge of the square if n=2, the green face of the cube if
        n=3, ...) by increasing and reducing his velocity in the n directions.

    Observation: 
        Type: Box(2n)
        Num     Observation                         Min     Max
        0       Agent 1st direction position        -1      1
        1       Agent 2nd direction position        -1      1
        ...     ...                                 ...     ...
        n-1     Agent nth direction position        -1      1
        n       Agent 1st direction velocity        -0.1    0.1
        ...     ...                                 ...     ...
        2n-2    Agent (n-1)th direction velocity    -0.1    0.1
        2n-1    Agent nth direction velocity        -0.1    0.1

    Actions:
        Type: Discrete(2n+1)
        Num     Action
        0       Don't Accelerate the agent
        1       Accelerate the agent to -1 orientation in the 1st direction
        2       Accelerate the agent to 1 orientation in the 1st direction
        3       Accelerate the agent to -1 orientation in the 2nd direction
        4       Accelerate the agent to 1 orientation in the 2nd  direction
        ...     ...
        2n-1    Accelerate the agent to -1 orientation in the nth direction
        2n      Accelerate the agent to 1 orientation in the nth direction

        
    Reward:
        Reward is 1 if the agent find the green (n-1)-face
        Raward is 0.1 if the agent find a red (n-1)-face

    Starting State
        Agent 1st direction position is assigned a uniform random value between -0.6 and-0.4
        Agent 2nd direction position is assigned to 0
        ...
        Agent nth direction position is assigned to 0
        Agent 1st direction velocity is assigned to 0
        Agent 2nd direction velocity is assigned to 0
        ...
        Agent nth direction velocity is assigned to 0

    Episode Termination:
        Agent find any (n-1)-face
        Episode length is greater than 100n
    """

    metadata = {
            'render.modes': ['human','rgb_array'],
            'video.frames_per_second': 30
            }


    def __init__(self, n_dimensions=1):

        if n_dimensions <= 0:
            raise ValueError('Number of dimension must be strictly positive')

        self.n = n_dimensions

        self.max_position = 1
        self.max_speed = 0.1
        
        self.high = np.array([
            np.ones((self.n))*self.max_position,
            np.ones((self.n))*self.max_speed
            ])
        self.low = -self.high

        self.action_space = spaces.Discrete(2*self.n+1)
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

        #action -> (direction,orientation)
        if action==0:
            orientation = 1
            direction = -1
        else:
            action-=1
            if action%2==0:
                orientation = -1
            elif action%2==1:
                orientation = 1
            direction = action//2
        
        #maj position and velocity
        position = self.state[0]
        velocity = self.state[1]
        if direction > -1:
            velocity[direction] += (orientation)*0.01
            for direction in range(self.n):
                if velocity[direction]>0:
                    velocity[direction] = max(0,velocity[direction]-0.001)
                else:
                    velocity[direction] = min(0,velocity[direction]+0.001)
            velocity[direction] = np.clip(velocity[direction], -self.max_speed, self.max_speed)
            position += velocity
            for direction in range(self.n):
                position[direction] = np.clip(position[direction], -self.max_position, self.max_position)

        #door
        green_door = bool(position[0] >= self.max_position)
        red_door = bool(position[0] <= -self.max_position)
        for direction in range(1,self.n):
            red_door = red_door or bool(position[direction] <= -self.max_position)
            red_door = red_door or bool(position[direction] >= self.max_position)
        done = bool(green_door or red_door)
        if green_door:
            reward = 1
            info = "green door"
        elif red_door:
            reward =  0.1
            info = "red door"
        else:
            reward = -0.01
            info = ""

        self.state = (position,velocity)
        return np.array(self.state), reward, done, info


    def reset(self):
        self.state = np.array([
            np.zeros((self.n)),
            np.zeros((self.n))
            ])
        self.state[0,0] = self.np_random.uniform(low=-0.6, high=-0.4)
        return np.array(self.state)


    def render(self, mode='human'):

        if self.n >= 3:
            return None

        if self.n==1:
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
                self.greenflag = rendering.FilledPolygon([
                    (greenflagx1, greenflagy1),
                    (greenflagx2, greenflagy1),
                    (greenflagx2, greenflagy2),
                    (greenflagx1, greenflagy2)
                    ])
                self.greenflag.set_color(0,1,0)
                self.viewer.add_geom(self.greenflag)

                #redflag
                redflagx1 = 0
                redflagx2 = 10
                redflagy1 = tracky+20
                redflagy2 = tracky-20
                self.redflag = rendering.FilledPolygon([
                    (redflagx1, redflagy1),
                    (redflagx2, redflagy1),
                    (redflagx2, redflagy2),
                    (redflagx1, redflagy2)
                    ])
                self.redflag.set_color(1,0,0)
                self.viewer.add_geom(self.redflag)
                
            if self.state is None:
                return None

            pos = self.state[0]
            agentx = (pos+self.max_position)*scale
            self.agenttrans.set_translation(agentx,agenty)

        elif self.n==2:
            screen_width = 600
            screen_height = 600

            world_width = self.max_position*2
            scale = screen_width/world_width

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(screen_width, screen_height)

                #reddoor
                reddoorx1 = 0
                reddoorx2 = screen_width
                reddoory1 = screen_height
                reddoory2 = screen_height-10
                self.reddoor = rendering.FilledPolygon([
                    (reddoorx1, reddoory1),
                    (reddoorx2, reddoory1),
                    (reddoorx2, reddoory2),
                    (reddoorx1, reddoory2)
                    ])
                self.reddoor.set_color(1,0,0)
                self.viewer.add_geom(self.reddoor)
                
                reddoorx1 = 0
                reddoorx2 = 10
                reddoory1 = screen_height
                reddoory2 = 0
                self.reddoor = rendering.FilledPolygon([
                    (reddoorx1, reddoory1),
                    (reddoorx2, reddoory1),
                    (reddoorx2, reddoory2),
                    (reddoorx1, reddoory2)
                    ])
                self.reddoor.set_color(1,0,0)
                self.viewer.add_geom(self.reddoor)
                
                reddoorx1 = 0
                reddoorx2 = screen_width
                reddoory1 = 0
                reddoory2 = 10
                self.reddoor = rendering.FilledPolygon([
                    (reddoorx1, reddoory1),
                    (reddoorx2, reddoory1),
                    (reddoorx2, reddoory2),
                    (reddoorx1, reddoory2)
                    ])
                self.reddoor.set_color(1,0,0)
                self.viewer.add_geom(self.reddoor)
                
                #greendoor
                greendoorx1 = screen_width-10
                greendoorx2 = screen_width
                greendoory1 = 0
                greendoory2 = screen_height
                self.greendoor = rendering.FilledPolygon([
                    (greendoorx1, greendoory1),
                    (greendoorx2, greendoory1),
                    (greendoorx2, greendoory2),
                    (greendoorx1, greendoory2)
                    ])
                self.greendoor.set_color(0,1,0)
                self.viewer.add_geom(self.greendoor)
                
                #agent
                agentwidth = 20
                agentheight = 20
                l,r,t,b = (-agentwidth/2, agentwidth/2, agentheight/2,
                        -agentheight/2)
                self.agent = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
                self.agenttrans = rendering.Transform()
                self.agent.add_attr(self.agenttrans)
                
                self.viewer.add_geom(self.agent)
                
            if self.state is None: return None

            pos = self.state[0]
            agent_position = (pos+self.max_position)*scale
            self.agenttrans.set_translation(agent_position[0],agent_position[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
