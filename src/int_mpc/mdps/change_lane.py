import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (Any, Dict, List, MutableSequence, Optional, Sequence,
                    Tuple, Union)

import numpy as np
from dataclasses_json import dataclass_json
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import LaneIndex
from highway_env.vehicle.kinematics import Vehicle
from overrides.overrides import overrides

from mdps.mdp_abc import (Action, DiscreteEnvironment, Metrics, PolicyBase,
                          State)

from . import highway_mdp
from .highway_mdp import HighwayEnvDiscreteAction, HighwayEnvState


@dataclass_json
@dataclass
class ChangeLaneConfig:
    lanes_count: int = field(default=4)
    vehicles_count: int = field(default=100)
    initial_spacing: float = field(default=1.0)
    max_episode_steps: int = field(default=30)
    alpha: float = field(default=0.4)
    beta: float = field(default=-1.0)
    reward_speed_range: Tuple[float, float] = field(
        default_factory=lambda: (20.0, 30.0))


@dataclass
class ChangeLaneMetrics(Metrics):
    total_reward: float = field()
    distance_travel: float = field()
    terminated_crash: bool = field()
    n_steps_to_crash: int = field()
    screenshot: Optional[np.ndarray] = field(default=None)


class ChangeLaneEnv(DiscreteEnvironment):
    class ChangeLaneRandomPolicy(PolicyBase):
        def __init__(self):
            super().__init__()

        @overrides
        def sample_action(self, state: State) -> Action:
            all_actions = list(HighwayEnvDiscreteAction)
            action: HighwayEnvDiscreteAction = random.choice(all_actions)
            return action

    StateType = HighwayEnvState
    ActionType = HighwayEnvDiscreteAction
    MetricsType = ChangeLaneMetrics

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
        "high_speed_reward": 0.4,
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
    N_ACTIONS: int = len(HighwayEnvDiscreteAction)
    STATES_SHAPE: np.ndarray = np.array([7, 4], dtype=np.uint)

    # protected instance variables
    _env: HighwayEnv
    _config: Dict[str, Any]
    _total_steps: int
    _start_state: Union[HighwayEnvState, None]
    _end_state: Union[HighwayEnvState, None]
    _reward_hist: MutableSequence[float]
    _DEFAULT_ACTION: Optional[HighwayEnvDiscreteAction]

    # public methods
    def __init__(
            self,
            lanes_count: int = 4,
            vehicles_count: int = 50,
            initial_spacing: float = 1.0,
            alpha: float = 0.4,
            beta: float = -1.0,
            reward_speed_range: Tuple[float, float] = (20, 30),
            default_action: Optional[HighwayEnvDiscreteAction] = None) -> None:
        """Constructor for ChangeLaneEnv

        Args:
            lanes_count (int, optional): Number of lanes in the environment. Defaults to 4.
            vehicles_count (int, optional): The total number of vehicles in the environment. Defaults to 50.
            initial_spacing (float, optional): The initial spacing between vehicles. Defaults to 1.
            alpha (float, optional): The magnitude to encourge fast speed. Defaults to 0.4.
            beta (float, optional): The reward signal when having collision with other vehicles. Should be negative real number. Defaults to -1.
            reward_speed_range (Tuple[float, float], optional): The range of speed that is encouraged. Defaults to (20, 30).
            default_action (Optional[HighwayEnvDiscreteAction]): The default action to take when the action passed in is unavailable.
        """
        super().__init__()
        self._config = ChangeLaneEnv._make_config(
            lanes_count=lanes_count,
            vehicles_count=vehicles_count,
            initial_spacing=initial_spacing,
            alpha=alpha,
            beta=beta,
            reward_speed_range=reward_speed_range)
        self._env = highway_mdp.make_highway_env(config=self._config)
        self._total_steps = 0
        self._start_state = None
        self._end_state = None
        self._reward_hist = list()
        self._DEFAULT_ACTION = default_action

    @overrides
    def step(
            self,
            action: Action,
            to_visualize: bool = False) -> Tuple[HighwayEnvState, float, bool]:
        """Take an action.

        Args:
            action (Action): The action to be taken.
            to_visualize (bool): Whether to render the visualization.

        Raises:
            TypeError: action not type of HighwayEnvDiscreteAction
            ValueError: action is invalid.

        Returns:
            mdp_state (State): The next state after taking the passed in action.
            reward (float): The reward associated with the state.
            is_terminal (bool): Whether or not the state is terminal.
        """
        if not isinstance(action, HighwayEnvDiscreteAction):
            raise TypeError
        if action not in HighwayEnvDiscreteAction:
            raise ValueError
        # if default action is present and the current action is unavailable
        if (self._DEFAULT_ACTION is not None) and \
            (action not in self._env.get_available_actions()):
            action = self._DEFAULT_ACTION
        _, reward, is_terminal, info = self._env.step(action=action)
        observation: np.ndarray = self._make_observation()
        # info = {'speed': 29.1455588268693, 'crashed': False, 'action': 3, 'cost': 0.0}
        mdp_state = HighwayEnvState(observation=observation,
                                    speed=info["speed"],
                                    is_crashed=info["crashed"],
                                    cost=info["cost"])
        self._total_steps = self._total_steps + 1
        self._end_state = deepcopy(mdp_state)
        if to_visualize:
            self._env.render()
        return mdp_state, reward, is_terminal

    @overrides
    def reset(self) -> HighwayEnvState:
        """Reset the environment

        Returns:
            mdp_state (State): The next state after taking the passed in action.
        """
        self._env.close()
        _ = self._env.reset()
        self._reward_hist.clear()
        observation: np.ndarray = self._make_observation()
        mdp_state = HighwayEnvState(observation=observation,
                                    speed=-1.0,
                                    is_crashed=False,
                                    cost=-1.0)
        self._total_steps = 0
        self._start_state = deepcopy(mdp_state)
        self._end_state = None
        return mdp_state

    @overrides
    def calculate_metrics(self) -> ChangeLaneMetrics:
        """Calculate the metrics for current episode.

        Raises:
            ValueError: When episode is not started.

        Returns:
            ChangeLaneMetrics: The metrics associated with current episode.
        """
        if self._start_state is None:
            raise ValueError("begin state initialized properly")
        if self._end_state is None:
            raise ValueError("end state not initialized")
        total_rewards: float = np.sum(self._reward_hist)
        start_loc: float = self._start_state.observation[0, 0]
        end_loc: float = self._end_state.observation[0, 0]
        distance_travel: float = end_loc - start_loc
        terminated_crash: bool = self._end_state.is_crashed
        n_steps_to_crash: int = self._total_steps if terminated_crash else -1
        screenshot: Union[np.ndarray, None] = self._env.render(
            mode="rgb_array") if terminated_crash else None
        metrics = ChangeLaneMetrics(total_reward=total_rewards,
                                    distance_travel=distance_travel,
                                    terminated_crash=terminated_crash,
                                    n_steps_to_crash=n_steps_to_crash,
                                    screenshot=screenshot)
        return metrics

    @overrides
    def summarize_metrics_seq(
            self, metrics_seq: Sequence[Metrics]) -> Dict[str, float]:
        """Summarize a sequence of Metrics.

        Calculate the sample mean and standard deviation for all the recorded value.

        Args:
            metrics_seq (Sequence[Metrics]): The sequence to be summarized.

        Raises:
            ValueError: If the passed in metrics is not supported.

        Returns:
            Dict[str, float]: The summarized sequence.
        """
        distance_travel: List[float] = list()
        steps_to_crash: List[int] = list()
        total_rewards: List[float] = list()
        for metric in metrics_seq:
            if isinstance(metric, ChangeLaneMetrics):
                distance_travel.append(metric.distance_travel)
                total_rewards.append(metric.total_reward)
                if metric.terminated_crash:
                    steps_to_crash.append(metric.n_steps_to_crash)
            else:
                raise ValueError
        avg_total_reward: float = np.mean(total_rewards)
        avg_distance: float = np.mean(distance_travel)
        avg_steps_to_crash: float = np.mean(steps_to_crash)
        std_total_reward: float = np.std(total_rewards)
        std_distance: float = np.std(distance_travel)
        std_steps_to_crash: float = np.std(steps_to_crash)
        n_crashes: int = len(steps_to_crash)
        metrics_dict: Dict[str, float] = {
            "avg_total_reward": avg_total_reward,
            "avg_distance": avg_distance,
            "avg_steps_to_crash": avg_steps_to_crash,
            "std_total_reward": std_total_reward,
            "std_distance": std_distance,
            "std_steps_to_crash": std_steps_to_crash,
            "n_crashes": n_crashes
        }
        return metrics_dict

    @overrides
    def get_random_policy(self) -> ChangeLaneRandomPolicy:
        return ChangeLaneEnv.ChangeLaneRandomPolicy()

    # protected methods
    def _make_observation(self) -> np.ndarray:
        """Make an observation based on the current vehicle state.

        Raises:
            ValueError: Raised when self._env.road.network is None or 

        Returns:
            obs (np.ndarray): (7, 4) The current observation of the ego vehicle. The shape corresponds to 7 vehicles in total and 4 statistics (x_pos, y_pos, x_vel, y_vel) of the vehicle.
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
        # add the lane that ego vehicle is currently in to index 0
        side_lanes.insert(0, lane_idx)
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
        if 1 <= len(obs_list) and len(obs_list) < 7:
            # add empty vehicle to the observation
            for _ in range(7 - len(obs_list)):
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
        self._reward_hist.append(env_reward)
        # currently use the system built-in reward formula
        return env_reward

    # protected static methods
    @staticmethod
    def _make_config(lanes_count: int, vehicles_count: int,
                     initial_spacing: float, alpha: float, beta: float,
                     reward_speed_range: Tuple[float, float]):
        config: Dict[str, Any] = deepcopy(ChangeLaneEnv.DEFAULT_CONFIG)
        config["lanes_count"] = lanes_count
        config["vehicles_count"] = vehicles_count
        config["initial_spacing"] = initial_spacing
        config["high_speed_reward"] = alpha
        config["collision_reward"] = beta
        config["reward_speed_range"] = reward_speed_range
        return config
