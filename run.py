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


rt_y = gym.make("rt-m-v0",render_mode = "rgb_array")
obs, info = rt_y.reset()


img = rt_y.render()
plt.imshow(img)
plt.show()