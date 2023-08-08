"""Custom Highway Environment"""
from typing import Dict, List

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import RoadNetwork, Road
from highway_env.road.lane import StraightLane, CircularLane, SineLane, LineType
import numpy as np

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle


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
                "vehicles_count": 8,
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
            "other_vehicles": 7,
            "screen_width": 1200,
            "screen_height": 800,
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
        """road_network: RoadNetwork = RoadNetwork()
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
        self.road = road"""
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 15, 15, 15, 15, 15, 15, 15, 15]

        # Initialise First Lane
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                            speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b",
                     StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                  speed_limit=speedlimits[1]))

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -20], [120, -30],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -20], [125, -30],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2 + 5, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3 + 5, np.deg2rad(0), np.deg2rad(136), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[6]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h",
                     CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4 + 5, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4 + 5, np.deg2rad(170), np.deg2rad(58), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a",
                     CircularLane(center5, radii5 + 5, np.deg2rad(240), np.deg2rad(270), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
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
            lane_index = self.road.network.random_lane_index(rng)

            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)


        # Other vehicles
        for i in range(self.config["other_vehicles"]):
            lane_indexes: list = [('i', 'a', 1), ('d', 'e', 1), ('h', 'i', 0), ('c', 'd', 1), ('g', 'h', 1), ('h', 'i', 0), ('b', 'c', 1), ('e', 'f', 0), ('d', 'e', 0)]
            random_lane_index = lane_indexes[i]
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=3 + rng.uniform(high=8))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)
