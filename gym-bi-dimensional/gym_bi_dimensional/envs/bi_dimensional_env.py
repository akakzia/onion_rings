import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class BiDimensionalEnv(gym.Env):
    """
    Description:
        An agent is in a square. The goal is to find the green door by
        increasing and reducing his velocity in the two directions.

    Observation: 
        Type: Box(4)
        Num Observation     Min     Max
        0   Agent X Position  -1      1
        1   Agent Y Position  -1      1
        2   Agent X Velocity  -0.1    0.1
        3   Agent Y Velocity  -0.1    0.1

    Actions:
        Type: Discrete(5)
        Num Action
        0   Don't Accelerate agent
        1   Accelerate agent to -1 orientation in X direction
        2   Accelerate agent to 1 orientation in X direction
        3   Accelerate agent to -1 orientation in Y direction
        4   Accelerate agent to 1 orientation in Y direction

        
    Reward:
        Reward is 1 if the agent find the green door
        Raward is 0.1 if the agent find the red door

    Starting State
        Agent X position is assigned a uniform random value between -0.6 and-0.4
        Agent Y position is 0
        Agent X velocity is assigned to 0
        Agent Y velocity is assigned to 0

    Episode Termination:
        Agent find any door
        Episode length is greater than 100
    """

    metadata = {
            'render.modes': ['human','rgb_array'],
            'video.frames_per_second': 30
            }


    def __init__(self):
        self.max_position = 1
        self.max_speed = 0.1
        
        self.high = np.array([self.max_position,self.max_position,
            self.max_speed,self.max_speed])
        self.low = -self.high

        self.action_space = spaces.Discrete(5)
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
            for direction in range(len(velocity)):
                if velocity[direction]>0:
                    velocity[direction] = max(0,velocity[direction]-0.001)
                else:
                    velocity[direction] = min(0,velocity[direction]+0.001)
            velocity[direction] = np.clip(velocity[direction], -self.max_speed, self.max_speed)
            position += velocity
            for direction in range(len(position)):
                position[direction] = np.clip(position[direction], -self.max_position, self.max_position)

        #door
        green_door = bool(position[0] >= self.max_position)
        red_door = bool(position[0] <= -self.max_position)
        for direction in range(1,len(position)):
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
            [self.np_random.uniform(low=-0.6, high=-0.4), 0],
            [0                                          , 0]])
        return np.array(self.state)


    def render(self, mode='human'):
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
            self.reddoor = rendering.FilledPolygon(
                    [(reddoorx1, reddoory1),
                        (reddoorx2, reddoory1),
                        (reddoorx2, reddoory2),
                        (reddoorx1, reddoory2)]
                    )
            self.reddoor.set_color(1,0,0)
            self.viewer.add_geom(self.reddoor)
            
            reddoorx1 = 0
            reddoorx2 = 10
            reddoory1 = screen_height
            reddoory2 = 0
            self.reddoor = rendering.FilledPolygon(
                    [(reddoorx1, reddoory1),
                        (reddoorx2, reddoory1),
                        (reddoorx2, reddoory2),
                        (reddoorx1, reddoory2)]
                    )
            self.reddoor.set_color(1,0,0)
            self.viewer.add_geom(self.reddoor)
            
            reddoorx1 = 0
            reddoorx2 = screen_width
            reddoory1 = 0
            reddoory2 = 10
            self.reddoor = rendering.FilledPolygon(
                    [(reddoorx1, reddoory1),
                        (reddoorx2, reddoory1),
                        (reddoorx2, reddoory2),
                        (reddoorx1, reddoory2)]
                    )
            self.reddoor.set_color(1,0,0)
            self.viewer.add_geom(self.reddoor)
            
            #greendoor
            greendoorx1 = screen_width-10
            greendoorx2 = screen_width
            greendoory1 = 0
            greendoory2 = screen_height
            self.greendoor = rendering.FilledPolygon(
                    [(greendoorx1, greendoory1),
                        (greendoorx2, greendoory1),
                        (greendoorx2, greendoory2),
                        (greendoorx1, greendoory2)]
                    )
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
