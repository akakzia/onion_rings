# RL-evaluation-environment 

This repository holds a gym compatible environment for RL algorithms benchmarking like :
 - DQN
 - DDPG
 - TD3

The environment is an hypercube in which the agent has to reach a specific hyperface to get a reward.

1D environement
:----------:
![1d](https://github.com/hroussille/RL-evaluation-environment/blob/master/visualizations/1d.png)

2D environement with rewards on all dimensions | 2D environement with rewards on the first dimension
:------:|:--------:
![2d 1 reward](https://github.com/hroussille/RL-evaluation-environment/blob/master/visualizations/2d_1reward.png) | ![2d 2 rewards](https://github.com/hroussille/RL-evaluation-environment/blob/master/visualizations/2d_2reward.png)

The visualizations represents the high reward of the environement in green and low reward in red.
environenment with 3D or more aren't visualizable but the works the same

## Installation

After cloning the repo , install the module :

```sh
cd gym-multi-dimensional
pip install -e .
```
## Usage

In order to provide a practical dimensions scaling we dynamically register the environment :

```python
import gym
import gym_multi_dimensional

id = gym_multi_dimensional.dynamic_register(n_dimensions=2,
        env_description={},continuous=True,acceleration=True)

env = gym.make(id)
```

## Options

 - n_dimensions : number of dimensions of the hyper rectangle
 - env_description : { "high_reward_value" : ,"low_reward_value" : , "high_reward_count" : , "low_rewars_count" : }
 - continuous : use continuous actions
 - acceleration : actions represents accelerations instead of velocity

## This reposity is under active development
