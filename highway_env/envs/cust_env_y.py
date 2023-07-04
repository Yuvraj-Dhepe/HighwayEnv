from itertools import repeat, product
from typing import Tuple, Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle


class CustEnvY(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
                "target_speeds": [0, 5, 10]
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,
            "collision_reward": -1,
            "lane_centering_cost": 4,
            "lane_centering_reward": 1,
            "action_reward": -0.3,
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "screen_width": 900,
            "screen_height": 600,
            "centering_position": [0.6, 0.92],
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1/(1+self.config["lane_centering_cost"]*lateral**2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 15,15, 15, 15, 15, 15, 15, 15]
        lane = StraightLane([0, 0], [60, 0], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([0, 5], [60, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))
        
        
        # 2 Circular Arc - 1
        center1 = [60,-20]
        radii1 = 20 
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-90), width=5, 
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+5, np.deg2rad(90), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))
        
        # 3 Straight Lane: R -> L
        net.add_lane("c", "d", StraightLane([60, -40], [20, -40],
                                                    line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                                                    speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([60, -45], [20, -45],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))
        
        # 4 Circular Arc - 2
        center2 = [20,-55]
        radii2 = 10 
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+5, np.deg2rad(90), np.deg2rad(270), width=5, 
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(90), np.deg2rad(270), width=5,
                                  clockwise=True, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))
        
        # 5 Straight Lane L -> R
        net.add_lane("e", "f", StraightLane([20, -70], [60, -70],
                                            line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                                            speed_limit=speedlimits[5]))
        net.add_lane("e", "f", StraightLane([20, -65], [60, -65],
                                                    line_types=(LineType.NONE, LineType.CONTINUOUS), width=5,
                                                    speed_limit=speedlimits[5]))
        
        # 6 Circular Arc - 3
        center3 = [60,-80]
        radii3 = 10 
        net.add_lane("f", "g",
                     CircularLane(center3, radii3, np.deg2rad(90), np.deg2rad(-90), width=5,  
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[6]))
        net.add_lane("f", "g",
                     CircularLane(center3, radii3+5, np.deg2rad(90), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[6]))
        
        # 7 Straight Lane
        net.add_lane("g", "h", StraightLane([60,-90], [0, -90],
                                                    line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                                    speed_limit=speedlimits[7]))
        net.add_lane("g", "h", StraightLane([60, -95], [0, -95],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[7]))
        
        # 8 Circular Arc - 4
        center4 = [0,-45]
        radii4 = 45 
        net.add_lane("h", "a",
                     CircularLane(center4, radii4, np.deg2rad(270), np.deg2rad(90), width=5, #angolo della curva 
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[8]))
        net.add_lane("h", "a",
                     CircularLane(center4, radii4+5, np.deg2rad(270), np.deg2rad(90), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6+rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6+rng.uniform(high=3))
            vehicle.randomize()
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)
