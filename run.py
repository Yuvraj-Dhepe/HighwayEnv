import numpy as np
from stable_baselines3 import DQN
# from highway_env import utils
# from highway_env.envs.common.abstract import AbstractEnv
# from highway_env.envs.common.action import Action
# from highway_env.road.road import Road, RoadNetwork
# from highway_env.utils import near_split
# from highway_env.vehicle.controller import ControlledVehicle
# from highway_env.vehicle.kinematics import Vehicle
# from gymnasium.envs.registration import register
# from gymnasium.envs import registry

from typing import Dict, Text
import gymnasium as gym
from highway_env.envs.cust_env_y import *
from highway_env.envs.cust_env_m import *
from highway_env.envs.simple_env import *
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.env_util import make_vec_env 
import os
import multiprocessing

LOAD = True
n_cpu = os.cpu_count()


env = gym.make("rt-y-v0",render_mode = "human")
obs, info = env.reset()
done = False
# env = make_vec_env('rt-y-v0',n_envs = n_cpu,vec_env_cls = SubprocVecEnv,seed = 7113)
# Rest settings for race-track are not changed, accept longitudinal = True, and Discrete Action being used
if LOAD == False:
	#print(env.action_space.sample())
	while not done:
		# print(env.get_available_actions())
		ac = env.action_space.sample()
		# ac = int(input())
		# env.step(ac)
		env.return_speed_and_velocity()
else:
	#obs, info = env.reset()
	model = DQN.load("y_models/DQN_models/mlp_dqn1.zip", env = env)
	while not done:
		ac = env.action_space.sample()
		env.step(ac)
		# env.return_speed_and_velocity()

# print(env.action_space)
# for i in range(1000):
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())