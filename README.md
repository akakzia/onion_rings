# RL-evaluation-environment 

This repository holds a gym compatible environment for RL algorithms benchmarking like :
 - DQN
 - DDPG
 - TD3

The environment is an hypercube in which the agent has to reach a specific hyperface to get a reward.

1D environment
:----------:
![1d](https://github.com/hroussille/RL-evaluation-environment/blob/master/visualizations/1d.png)

2D environment with rewards on all dimensions | 2D environement with rewards on the first dimension
:------:|:--------:
![2d 2 reward](https://github.com/hroussille/RL-evaluation-environment/blob/master/visualizations/2d_2reward.png) | ![2d 1 rewards](https://github.com/hroussille/RL-evaluation-environment/blob/master/visualizations/2d_1reward.png)

The visualizations represents the high reward of the environment in green and low reward in red.
Environments with 3 dimensions or more aren't visualizable but the works the same way.

## Installation

After cloning the repo , install the module :

```sh
cd gym-hypercube
pip install -e .
```
## Usage

In order to provide a practical dimensions scaling we dynamically register the environment :

```python
import gym
import gym_hypercube

id = gym_hypercube.dynamic_register(n_dimensions=2,
        env_description={'high_reward_value':1,
        'low_reward_value':0.1,
        'high_reward_count':'half',
        'low_reward_count':'half',
        'mode':'deterministic'},
        continuous=True,
        acceleration=True,
        reset_radius=None)

env = gym.make(id)
```

## Options

 - n_dimensions : number of dimensions of the hypercube
 - env_description : { "high_reward_value" : ,"low_reward_value" : , "high_reward_count" : , "low_rewars_count" : , "mode":  }
 - continuous : use continuous actions
 - acceleration : actions represents accelerations instead of velocity
 - reset_radius : respawn radius around the center of the environment

## This reposity is under active development
