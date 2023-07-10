import numpy as np

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
import matplotlib.pyplot as plt


env = gym.make("rt-ell-v0",render_mode = "human")
obs, info = env.reset()
# print(env.action_space)

for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
