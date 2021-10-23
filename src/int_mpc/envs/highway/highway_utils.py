from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict

import gym
import numpy as np
from highway_env.envs.highway_env import HighwayEnv

from ..env_abc import Action, State


class HighwayEnvDiscreteAction(Action, IntEnum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4
    INVALID = -1


@dataclass
class HighwayEnvState(State):
    observation: np.ndarray = field()
    speed: float = field()
    is_crashed: bool = field()
    prev_action: int = field()
    cost: float = field()


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
