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
        Reward is 0.1 if the agent find a red (n-1)-face

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
    """

    metadata = {
            'render.modes': ['human','rgb_array'],
            'video.frames_per_second': 30
            }

    """ Default environement description , to be customized by the user """
    default_description = { 'high_reward': 1 }


    def __init__(self, n_dimensions=1, env_description=default_description,continuous=True,acceleration=True):

        """ Sanity checks """
        if n_dimensions <= 0:
            raise ValueError('Number of dimension must be strictly positive')

#        if validate_env_description() is False:
#            raise ValueError("Missing descriptors in environement description")

        self.n = n_dimensions
        self.continuous = continuous
        self.acceleration = acceleration

        self.max_position = 1
        self.high_position = np.array([
            np.ones((self.n))*self.max_position,
            np.ones((self.n))*float('inf')
            ])
        self.low_position = -self.high_position
        self.observation_space = spaces.Box(low=self.low_position,
                high=self.high_position, dtype=np.float32)

        if self.continuous:
            self.max_action = 1
            self.high_action = np.ones((self.n))*self.max_action
            self.low_action = -self.high_action
            self.action_space = spaces.Box(low=self.low_action,
                    high=self.high_action, dtype=np.float32)

        else: #if discrete
            self.action_space = spaces.Discrete(2*self.n+1)

        if self.acceleration:
            self.power = 0.01
        else: #if velocity
            self.power = 0.1
        self.friction = 0.001
        self.accel = np.zeros((self.n))

        self.high_reward = 1
        self.low_reward = 0.1
        self.action_cost = 0.01
        
        self._max_episode_steps = 100
        
        self.viewer = None
        self.state = None
        self.seed()
        self.reset()

        """ Keep IDs of dimensions used for environement description """
        self.high_reward_position = []
        self.low_reward_position = []
        self.wall_position = []

        self.load_description(env_description)

    def _validate_env_description(self, env_description):
        return True
    
    def load_description(self, env_description):

        for i in range(self.n):

            """ Assign high boundary of nth dimension to high reward """
            self.high_reward_position.append((i, self.max_position))

            """ Assign low boundary of nth dimension to low reward """ 
            self.low_reward_position.append((i, - self.max_position))

            # self.wall_position.append((i, - self.max_position))

        print(self.high_reward_position)
        print(self.low_reward_position)



    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
                action, type(action))
        

        position = self.state[0]
        velocity = self.state[1]

        #update velocity
        if self.acceleration:
            if self.continuous:
                self.accel = action*self.power
                velocity += self.accel
            else: #if discrete
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
                if direction > -1:
                    self.accel = (orientation)*self.power
                    velocity[direction] += self.accel
        else: #if velocity
            if self.continuous:
                self.accel = action*self.power
                velocity = self.accel 
            else: #if discrete
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
                if direction > -1:
                    self.accel = (orientation)*self.power
                    velocity[direction] = self.accel

        #friction
        for direction in range(self.n):
            if velocity[direction]>0:
                velocity[direction] = max(0,velocity[direction]-self.friction)
            else:
                velocity[direction] = min(0,velocity[direction]+self.friction)
        
        #update position
        position += velocity
        for direction in range(self.n):
            position[direction] = np.clip(position[direction],
                    -self.max_position, self.max_position)

        reach_high_reward = False
        reach_low_reward = False

        """ Check for high reward in n dimensional space """
        for infos in self.high_reward_position:
            nth, boundary = infos
            if abs(position[nth] + boundary) >= 2 * self.max_position:
                    reach_high_reward = True

        """ Check for low reward in n dimensional space """
        for infos in self.low_reward_position:
            nth, boundary = infos
            if abs(position[nth] + boundary) >= 2 * self.max_position:
                    reach_low_reward = True

        """ Check for wall in n dimensional space """
        for infos in self.wall_position:
            nth, boundary = infos
            if abs(position[nth] + boundary) >= 2 * self.max_position:
                velocity[nth] = 0

        if reach_high_reward:
            reward = self.high_reward
            info = "high reward"
        elif reach_low_reward:
            reward =  self.low_reward
            info = "low reward"
        else:
            reward = -self.action_cost
            info = ""
        done = reach_high_reward or reach_low_reward

        self.state = [position,velocity]
        return np.array(self.state), reward, done, info


    def reset(self):
        self.state = np.array([
            np.random.uniform(low=-self.max_position, high=self.max_position,
                size=self.n),
            np.zeros((self.n))
            ])
        return self.state


    def get_vertices_list(self, nth, boundary, width, height, line=0):
        if self.n == 1:

            y1 = line + 20
            y2 = line - 20

            if nth == 0 and boundary == self.max_position:
                x1 = width - 10
                x2 = width

            if nth == 0 and boundary == -self.max_position:
                x1 = 0
                x2 = 10

            return x1, y1, x2, y2

        if self.n == 2:

            if nth == 0 and boundary == self.max_position:
                x1 = width - 10
                x2 = width
                y1 = 0
                y2 = height

            if nth == 0 and boundary == -self.max_position:

                x1 = 0
                x2 = 10
                y1 = 0
                y2 = height

            if nth == 1 and boundary == self.max_position:

                x1 = 0
                x2 = width
                y1 = height - 10
                y2 = height

            if nth == 1 and boundary == -self.max_position:

                x1 = 0
                x2 = width
                y1 = 10
                y2 = 0
            
            return x1, y1, x2, y2

        return None


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

                """ Line sprite """
                tracky = 100
                self.track = rendering.Line((0,tracky),(screen_width,tracky))
                self.track.set_color(0,0,0)
                self.viewer.add_geom(self.track)

                """ Agent sprite """
                agenty = tracky
                self.agent = rendering.make_circle(radius=15,res=30,filled=True)
                self.agenttrans = rendering.Transform()
                self.agent.add_attr(self.agenttrans)
                self.viewer.add_geom(self.agent)
                self.agentline = rendering.FilledPolygon([
                    (0, 0),(0, 2),(15, 2),(15, -2),(0,-2)
                    ])
                self.agentline.set_color(1,1,1)
                self.agentlinetrans = rendering.Transform()
                self.agentline.add_attr(self.agentlinetrans)
                self.viewer.add_geom(self.agentline)


                """ Build high reward sprite """
                for reward in self.high_reward_position:
                    nth, boundary = reward
                    x1, y1, x2, y2 = self.get_vertices_list(nth, boundary, screen_width, screen_height, line=tracky)
                    sprite = rendering.FilledPolygon([(x1, y1),(x2, y1),(x2, y2),(x1, y2)])
                    sprite.set_color(0,1,0)
                    self.viewer.add_geom(sprite)

                """ Build low reward sprite """
                for reward in self.low_reward_position:
                    nth, boundary = reward
                    x1, y1, x2, y2 = self.get_vertices_list(nth, boundary, screen_width, screen_height, line=tracky)
                    sprite = rendering.FilledPolygon([(x1, y1),(x2, y1),(x2, y2),(x1, y2)])
                    sprite.set_color(1,0,0)
                    self.viewer.add_geom(sprite)

                """ Wall sprite """
                
            if self.state is None:
                return None

            pos = self.state[0]
            vel = self.state[1]
            agentx = (pos+self.max_position)*scale
            self.agenttrans.set_translation(agentx,agenty)
            self.agentlinetrans.set_translation(agentx,agenty)
            self.agentlinetrans.set_rotation(math.atan2(0,vel))


        elif self.n==2:
            screen_width = 600
            screen_height = 600

            world_width = self.max_position*2
            scale = screen_width/world_width

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(screen_width, screen_height)
                
                """ Build high reward sprite """
                for reward in self.high_reward_position:
                    nth, boundary = reward
                    x1, y1, x2, y2 = self.get_vertices_list(nth, boundary, screen_width, screen_height)
                    sprite = rendering.FilledPolygon([(x1, y1),(x2, y1),(x2, y2),(x1, y2)])
                    sprite.set_color(0,1,0)
                    self.viewer.add_geom(sprite)

                """ Build low reward sprite """
                for reward in self.low_reward_position:
                    nth, boundary = reward
                    x1, y1, x2, y2 = self.get_vertices_list(nth, boundary, screen_width, screen_height)
                    sprite = rendering.FilledPolygon([(x1, y1),(x2, y1),(x2, y2),(x1, y2)])
                    sprite.set_color(1,0,0)
                    self.viewer.add_geom(sprite)

                """ Build wall sprite """

                """ Build agent sprite """
                self.agent = rendering.make_circle(radius=15,res=30,filled=True)
                self.agenttrans = rendering.Transform()
                self.agent.add_attr(self.agenttrans)
                self.viewer.add_geom(self.agent)
                self.agentline = rendering.FilledPolygon([
                    (0, 0),(0, 2),(15, 2),(15, -2),(0,-2)
                    ])
                self.agentline.set_color(1,1,1)
                self.agentlinetrans = rendering.Transform()
                self.agentline.add_attr(self.agentlinetrans)
                self.viewer.add_geom(self.agentline)
                
            if self.state is None: return None

            pos = self.state[0]
            vel = self.state[1]
            agent_position = (pos+self.max_position)*scale
            self.agenttrans.set_translation(agent_position[0],agent_position[1])
            self.agentlinetrans.set_translation(agent_position[0],agent_position[1])
            self.agentlinetrans.set_rotation(math.atan2(vel[1],vel[0]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
