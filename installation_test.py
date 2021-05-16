# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:14:57 2020

@author: Honghu Xue
"""

import matplotlib.pyplot as plt
import numpy as np

#------------------test gym classical control----------
import gym
gym.logger.set_level(gym.logger.ERROR)
env = gym.make('MountainCar-v0')
print('MountainCar\'s environment space', env.observation_space)
env.close()


#------------------test pytorch installation-------------------
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device) # return cuda if you have GPU accelration, else cpu.


#------------------test pybullet installation----------
import pybullet_envs
env = gym.make('HalfCheetahBulletEnv-v0')
print('Halfcheetah\'s environment space', env.observation_space)
env.close()

#------------------test gym box2d----------
env = gym.make('BipedalWalkerHardcore-v3')
print('BipedalWalkerHardcore\'s environment space', env.observation_space)
env.close()


#------------------test gym Atari----------
env = gym.make('Pong-v0')
print('Pong\'s environment space', env.observation_space)
env.close()


