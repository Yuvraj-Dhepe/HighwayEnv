"""Custom Highway Environment"""
from typing import Dict, List

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import RoadNetwork, Road
from highway_env.road.lane import StraightLane, CircularLane, SineLane, LineType
import numpy as np

from highway_env.vehicle.behavior import IDMVehicle


def make_nodes(single_nodes: List[str]) -> List[tuple]:
    shifted_nodes = single_nodes[1:]
    shifted_nodes += single_nodes[0]
    return [(node, next_node) for node, next_node in zip(single_nodes, shifted_nodes)]


class CustomRoadEnv(AbstractEnv):

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
                "type": "DiscreteAction",
                "longitudinal": False,
                "lateral": True,
                "target_speeds": [0, 2, 4]
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
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray):
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def get_config(self) -> dict:
        """
        Return the default config
        """
        return self.default_config()

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self):
        road_network: RoadNetwork = RoadNetwork()
        single_nodes: List[str] = ["a", "b", "c", "d", "e", "f", "g"]
        nodes: List[tuple] = make_nodes(single_nodes=single_nodes)
        speed: int = 10
        lane_width: int = 5
        cont_line: LineType = LineType.CONTINUOUS
        stripped_line: LineType = LineType.STRIPED
        none_line: LineType = LineType.NONE

        # initialize the first lane
        start: List[int] = [42, 0]
        end: List[int] = [100, 0]


        road_network.add_lane(nodes[0][0], nodes[0][1],
                              StraightLane(start=start, end=end, line_types=(cont_line, none_line), width=lane_width,
                                           speed_limit=speed))
        road_network.add_lane(nodes[0][0], nodes[0][1],
                              StraightLane([42, 5], [100, 5], line_types=(stripped_line, cont_line), width=lane_width,
                                           speed_limit=speed))


        # 2 - Circular Arc #1
        center1 = [100, -20]
        radius_1 = 20
        road_network.add_lane(nodes[1][0], nodes[1][1],
                              CircularLane(center1, radius_1, np.deg2rad(90), np.deg2rad(0), width=lane_width,
                                           line_types=[cont_line, none_line], speed_limit=speed, clockwise=False, ))
        road_network.add_lane(nodes[1][0], nodes[1][1],
                              CircularLane(center1, radius_1 + 5, np.deg2rad(90), np.deg2rad(0), width=lane_width,
                                           line_types=[stripped_line, cont_line], speed_limit=speed, clockwise=False, ))

        # counter circular Arc
        center2 = [140, -20]
        radius_2 = 20

        #
        road_network.add_lane(nodes[2][0], nodes[2][1],
                              CircularLane(center2, radius_2, np.deg2rad(-180), np.deg2rad(-90), width=lane_width,
                                           line_types=[cont_line, stripped_line],
                                           speed_limit=speed, clockwise=True, ))
        road_network.add_lane(nodes[2][0], nodes[2][1],
                              CircularLane(center2, radius_2 - 5, np.deg2rad(-180), np.deg2rad(-90), width=lane_width,
                                           line_types=[none_line, cont_line],
                                           speed_limit=speed, clockwise=True, ))

        # 180 degree circular Arc
        center3 = [140, -70]
        radius_3 = 30
        road_network.add_lane(nodes[3][0], nodes[3][1],
                              CircularLane(center3, radius_3, np.deg2rad(90), np.deg2rad(-90), width=lane_width,
                                           clockwise=False, line_types=[cont_line, none_line], speed_limit=speed))
        road_network.add_lane(nodes[3][0], nodes[3][1],
                              CircularLane(center3, radius_3 + lane_width, np.deg2rad(90), np.deg2rad(-90),
                                           width=lane_width,
                                           clockwise=False, line_types=[stripped_line, cont_line], speed_limit=speed))

        # add sine lane
        pulsation: float = 0.2  # Smoothness of the wave
        phase: float = 0  # The starting direction of the wave
        amplitude: float = 3
        road_network.add_lane(nodes[4][0], nodes[4][1],
                              SineLane(
                                  start=[140, -100], end=[80, -80], amplitude=amplitude, phase=phase,
                                  pulsation=pulsation, width=lane_width, speed_limit=speed,
                                  line_types=[cont_line, stripped_line]
                              )
                              )
        road_network.add_lane(nodes[4][0], nodes[4][1],
                              SineLane(
                                  start=[140, -105], end=[80, -85], amplitude=amplitude, phase=phase,
                                  pulsation=pulsation,
                                  width=lane_width, speed_limit=speed, line_types=[none_line, cont_line]))

        # add straight
        road_network.add_lane(nodes[5][0], nodes[5][1],
                              StraightLane([80, -80], [42, -80], line_types=(cont_line, stripped_line),
                                           width=lane_width,
                                           speed_limit=speed)
                              )
        road_network.add_lane(nodes[5][0], nodes[5][1],
                              StraightLane([80, -85], [42, -85], line_types=(none_line, cont_line),
                                           width=lane_width,
                                           speed_limit=speed))

        # 180 degree circular Arc
        center_4 = [42, -40]
        radius4 = 40
        road_network.add_lane(nodes[6][0], nodes[6][1],
                              CircularLane(center_4, radius4, np.deg2rad(270), np.deg2rad(90), width=lane_width,
                                           clockwise=False, line_types=[cont_line, none_line],
                                           speed_limit=speed))
        road_network.add_lane(nodes[6][0], nodes[6][1],
                              CircularLane(center_4, radius4 + 5, np.deg2rad(270), np.deg2rad(90), width=lane_width,
                                           clockwise=False, line_types=[stripped_line, cont_line],
                                           speed_limit=speed))

        road = Road(network=road_network, record_history=self.config["show_trajectories"])
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
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=0 + rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=0 + rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)
