from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from overrides import overrides

from mdps.mdp_abc import DiscreteAction, State


class HighwayEnvDiscreteAction(DiscreteAction):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4

    @staticmethod
    def _str_to_action(
            s: Optional[str]) -> Union["HighwayEnvDiscreteAction", None]:
        if s is None:
            return s
        try:
            return HighwayEnvDiscreteAction(int(s))
        except ValueError:
            return HighwayEnvDiscreteAction[s.upper()]


@dataclass
class HighwayEnvState(State):
    observation: np.ndarray = field()
    speed: float = field()
    is_crashed: bool = field()
    cost: float = field()

    @overrides
    def get_np_state(self, copy: bool = True) -> np.ndarray:
        if copy == True:
            return np.copy(self.observation)
        return self.observation


def make_highway_env(config: Dict[str, Any]) -> HighwayEnv:
    """Given a highway_env config, create a highway_env environment.

    A wrapper function that provides better Python type hint for HighwayEnv.

    Args:
        config (Dict[str, Any]): The configuration

    Returns:
        HighwayEnv: The new HighwayEnv.
    """
    env: HighwayEnv = gym.make("highway-v0")
    env.configure(config=config)
    return env
