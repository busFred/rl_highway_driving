import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import LaneIndex
from highway_env.vehicle.kinematics import Vehicle
from overrides import overrides

from ..mdp_abc import DiscreteEnvironment, State
from . import highway_mdp
from .highway_mdp import HighwayEnvDiscreteAction, HighwayEnvState


class ChangeLaneEnv(DiscreteEnvironment):

    # static const
    DEFAULT_CONFIG: Dict[str, Any] = {
        "observation": {
            "type": "Kinematics",
            "features": ['presence', 'x', 'y', 'vx', 'vy'],
            "normalize": False,
            "observe_intentions": False,
            "order": "sorted"
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 40,  # [s]
        "initial_spacing": 2,
        # The reward received when colliding with a vehicle.
        "collision_reward": -1,
        # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
        "reward_speed_range": [20, 30],
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,  # [px]
        "screen_height": 150,  # [px]
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }
    EMPTY_VEHICLE: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0])

    # protected
    _env: HighwayEnv
    _config: Dict[str, Any]

    def __init__(
        self,
        lanes_count: int = 4,
        vehicles_count: int = 50,
        initial_spacing: float = 1,
        reward_speed_range: Tuple[float, float] = (20, 30)
    ) -> None:
        super().__init__()
        self._config = ChangeLaneEnv._make_config(
            lanes_count=lanes_count,
            vehicles_count=vehicles_count,
            initial_spacing=initial_spacing,
            reward_speed_range=reward_speed_range)
        self._env = highway_mdp.make_highway_env(config=self._config)

    # @overrides
    def step(
            self,
            action: HighwayEnvDiscreteAction,
            to_visualize: bool = False) -> Tuple[HighwayEnvState, float, bool]:
        """Take an action.

        Args:
            action (Action): The action to be taken.
            to_visualize (bool): Whether to render the visualization.

        Raises:
            ValueError: action is invalid.

        Returns:
            mdp_state (State): The next state after taking the passed in action.
            reward (float): The reward associated with the state.
            is_terminal (bool): Whether or not the state is terminal.
        """
        if action not in HighwayEnvDiscreteAction:
            raise ValueError
        _, reward, is_terminal, info = self._env.step(action=action)
        observation: np.ndarray = self._make_observation()
        # info = {'speed': 29.1455588268693, 'crashed': False, 'action': 3, 'cost': 0.0}
        mdp_state = HighwayEnvState(observation=observation,
                                    speed=info["speed"],
                                    is_crashed=info["crashed"],
                                    cost=info["cost"])
        if to_visualize:
            self._env.render()
        return mdp_state, reward, is_terminal

    # @overrides
    def step_random(
        self,
        to_visualize: bool = False
    ) -> Tuple[HighwayEnvDiscreteAction, State, float, bool]:
        """Take a random action.

        The action ~ multinomial(n=1, p_vals=[1/5]*5).

        Args:
            to_visualize (bool): Whether to render the visualization.


        Returns:
            action (HighwayEnvDiscreteAction): The action being taken.
            mdp_state (State): The next state after taking the passed in action.
            reward (float): The reward associated with the state.
            is_terminal (bool): Whether or not the state is terminal.
        """
        all_actions = list(HighwayEnvDiscreteAction)
        action: HighwayEnvDiscreteAction = random.choice(all_actions)
        mdp_state, reward, is_terminal = self.step(action, to_visualize)
        return action, mdp_state, reward, is_terminal

    @overrides
    def reset(self) -> State:
        """Reset the environment

        Returns:
            mdp_state (State): The next state after taking the passed in action.
        """
        _ = self._env.reset()
        observation: np.ndarray = self._make_observation()
        mdp_state = HighwayEnvState(observation=observation,
                                    speed=-1.0,
                                    is_crashed=False,
                                    cost=-1.0)
        return mdp_state

    @overrides
    def int_to_action(self, action: int) -> HighwayEnvDiscreteAction:
        try:
            act = HighwayEnvDiscreteAction(action)
            return act
        except:
            raise ValueError("unsupporeted int action")

    # protected method
    def _make_observation(self) -> np.ndarray:
        """Make an observation based on the current vehicle state.

        Raises:
            ValueError: Raised when self._env.road.network is None or 

        Returns:
            obs (np.ndarray): (5, 4) The current observation of the ego vehicle. The shape corresponds to 5 vehicles in total and 4 statistics of the vehicle.
        """
        # env is not properly constructed
        if self._env.road.network is None:
            raise ValueError
        ego_v: Vehicle = self._env.vehicle
        obs_list: List[np.ndarray] = list()
        # ego_state = [pos_x, pos_y, vel_x, vel_y]
        ego_state: np.ndarray = self._make_vehicle_state(
            ego_v.position, ego_v.velocity)
        obs_list.append(ego_state)
        # get the index of the lane that ego is currently in
        lane_idx: LaneIndex = self._env.road.network.get_closest_lane_index(
            position=ego_v.position)
        # get all the lane next to the ego vehicle's current lane
        # max length of side_lanes is 2
        side_lanes: List[LaneIndex] = self._env.road.network.side_lanes(
            lane_index=lane_idx)
        for side_lane in side_lanes:
            # get leader and follower
            leader_v, follower_v = self._env.road.neighbour_vehicles(
                vehicle=ego_v, lane_index=side_lane)
            # calculate their relative state w.r.t ego_vehicle
            leader_state: np.ndarray = self._calculate_relative_state(
                ego_v, leader_v)
            follower_state: np.ndarray = self._calculate_relative_state(
                ego_v, follower_v)
            obs_list.append(leader_state)
            obs_list.append(follower_state)
        # catch the case where there is only one side lane
        if 1 <= len(obs_list) and len(obs_list) < 5:
            # add empty vehicle to the observation
            for _ in range(5 - len(obs_list)):
                obs_list.append(np.copy(ChangeLaneEnv.EMPTY_VEHICLE))
        obs: np.ndarray = np.array(obs_list)
        return obs

    def _make_vehicle_state(self, pos: np.ndarray,
                            vel: np.ndarray) -> np.ndarray:
        """Make vehicle state.

        Args:
            pos (np.ndarray): (2, ) The position of the vehicle.
            vel (np.ndarray): (2, ) The velocity of the vehicle.

        Returns:
            state (np.ndarray): (4, ) The state of the vehicle.
        """
        # concatenate position and velocity into a single observation vector
        state: np.ndarray = np.concatenate((pos, vel), axis=0)
        return state

    def _calculate_relative_state(
            self, ego_v: Vehicle, target_v: Union[Vehicle,
                                                  None]) -> np.ndarray:
        """Calculate relative vehicle state.

        Args:
            ego_v (Vehicle): The ego vehicle.
            target_v (Union[Vehicle, None]): The target vehicle.

        Returns:
            target_state (np.ndarray): (4, ) The target_v state relative to ego_v.
        """
        if target_v is None:
            return np.copy(ChangeLaneEnv.EMPTY_VEHICLE)
        # get relative position and velocity w.r.t ego vehicle
        rel_pos: np.ndarray = target_v.position - ego_v.position
        rel_vel: np.ndarray = target_v.velocity - ego_v.velocity
        # create the target state
        target_state: np.ndarray = self._make_vehicle_state(rel_pos, rel_vel)
        return target_state

    def _calculate_reward(self, state: np.ndarray, env_reward: float) -> float:
        """Calculate the reward.

        Args:
            state (np.ndarray): The state of the self._env
            env_reward (float): The reward given by self._env.step()

        Returns:
            reward (float): The current reward.
        """
        # currently use the system built-in reward formula
        return env_reward

    # protected static method
    @staticmethod
    def _make_config(lanes_count: int, vehicles_count: int,
                     initial_spacing: float, reward_speed_range: Tuple[float,
                                                                       float]):
        config: Dict[str, Any] = deepcopy(ChangeLaneEnv.DEFAULT_CONFIG)
        config["lanes_count"] = lanes_count
        config["vehicles_count"] = vehicles_count
        config["initial_spacing"] = initial_spacing
        config["reward_speed_range"] = reward_speed_range
        return config
