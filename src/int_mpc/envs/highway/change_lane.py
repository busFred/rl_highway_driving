from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Tuple

import gym
import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from overrides import overrides

from ..env_abc import Action, Environment, State


class ChangeLaneAction(Action, IntEnum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


@dataclass
class ChangeLaneState(State):
    state: np.ndarray = field()
    speed: float = field()
    is_crashed: bool = field()
    prev_action: int = field()
    cost: float = field()


class ChangeLane(Environment):

    # static const
    DEFAULT_CONFIG: Dict[str, Any] = {
        "observation": {
            "type": "Kinematics",
            "features": ['presence', 'x', 'y', 'vx', 'vy'],
            "normalize": False,
            "observe_intentions": False
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
        env: HighwayEnv = gym.make("highway-v0")
        config = deepcopy(ChangeLane.DEFAULT_CONFIG)
        config["lanes_count"] = lanes_count
        config["vehicles_count"] = vehicles_count
        config["initial_spacing"] = initial_spacing
        config["reward_speed_range"] = reward_speed_range
        env.configure(config=config)
        self._env = env
        self._config = config

    @overrides
    def step(self, action: ChangeLaneAction) -> Tuple[State, float, bool]:
        """Take an action.

        Args:
            action (Action): The action to be taken.

        Returns:
            state (State): The next state after taking the passed in action.
            reward (float): The reward associated with the state.
            is_terminal (bool): Whether or not the state is terminal.
        """
        obs, reward, is_terminal, info = self._env.step(action=action)
        # info = {'speed': 29.1455588268693, 'crashed': False, 'action': 3, 'cost': 0.0}
        state = ChangeLaneState(state=obs,
                                speed=info["speed"],
                                is_crashed=info["crashed"],
                                prev_action=info["action"],
                                cost=info["cost"])
        return state, reward, is_terminal

    @overrides
    def reset(self) -> State:
        """Reset the environment

        Returns:
            state (State): Returns the state of the new environment.
        """
        obs = self._env.reset()
        state = ChangeLaneState(state=obs,
                                speed=-1.0,
                                is_crashed=False,
                                prev_action=-1,
                                cost=-1.0)
        return state
