from highway_env.envs.highway_env import *
from highway_env.envs.merge_env import *
from highway_env.envs.parking_env import *
from highway_env.envs.roundabout_env import *
from highway_env.envs.two_way_env import *
from highway_env.envs.intersection_env import *
from highway_env.envs.lane_keeping_env import *
from highway_env.envs.u_turn_env import *
from highway_env.envs.exit_env import *
from highway_env.envs.racetrack_env import *
from highway_env.envs.cust_env_m import *
from highway_env.envs.cust_env_y import *
from gymnasium.envs.registration import register
from highway_env.envs.basic_env import *

register(
    id='rt-m-v0',
    entry_point='highway_env.envs:CustEnvM'
)

register(
    id='rt-y-v0',
    entry_point='highway_env.envs:CustEnvY'
)

register(
    id='rt-ell-v0',
    entry_point='highway_env.envs:EllipseEnv'
)