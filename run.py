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
from highway_env.envs.simple_env import *
import matplotlib.pyplot as plt


env = gym.make("racetrack-v0",render_mode = "human")
# Rest settings for race-track are not changed, accept longitudinal = True, and Discrete Action being used
env.configure({
    "manual_control": True
})
obs, info = env.reset()
done = False
print(env.action_space.sample())
while not done:
    ac = int(input())
    # print(env.get_available_actions())
    # ac = env.action_space.sample()
    env.step(ac)

# print(env.action_space)
# for i in range(1000):
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
