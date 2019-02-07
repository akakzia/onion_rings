# RL-evaluation-environment 

This repository holds a gym compatible environment for RL algorithms benchmarking like :
 - DDPG
 - TD3

This environment is an hyperrectangle in which the agent has to reach a specific side/face to get a reward.

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
 - env_description : NOT YET IMPLEMENTED
 - continuous : use continuous actions
 - acceleration : actions represents accelerations instead of velocity

## This reposity is under active development
