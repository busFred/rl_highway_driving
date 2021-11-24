from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Sequence, Tuple

import numpy as np
from overrides.overrides import overrides


class Action(ABC):
    @abstractmethod
    def get_np_action(self, copy=True) -> np.ndarray:
        """Get np action

        Args:
            copy (bool, optional): Makes the value independent from self. Defaults to True.

        Returns:
            action (np.ndarray): (action_dim, ) The numpy representation of the action.
        """
        pass


class _DiscreteAction(type(Action), type(IntEnum)):
    pass


class DiscreteAction(Action, IntEnum, metaclass=_DiscreteAction):
    @overrides
    def get_np_action(self, copy=True) -> np.ndarray:
        return np.array([self.value], copy=copy)


class ContinuousAction(Action):
    pass


class State(ABC):
    @abstractmethod
    def get_np_state(self, copy: bool = True) -> np.ndarray:
        """Get numpy representation of  state.

        Args:
            copy (bool, optional): Whether to copy the data or not. Defaults to True.

        Returns:
            state_np (np.ndarray): The numpy representation of the state.
        """
        pass


@dataclass
class Metrics(ABC):
    pass


class PolicyBase(ABC):

    ActionType = Action

    @abstractmethod
    def sample_action(self, state: State) -> Action:
        pass


class Environment(ABC):

    StateType = State
    ActionType = Action
    MetricsType = Metrics

    @abstractmethod
    def step(self,
             action: Action,
             to_visualize: bool = False) -> Tuple[Action, State, float, bool]:
        """Take an action.

        Args:
            action (Action): The action to be taken.
            to_visualize (bool): Whether to render the visualization.

        Returns:
            state (State): The next state after taking the passed in action.
            action (Action): The actual action being taken by the agent.
            reward (float): The reward associated with the state.
            is_terminal (bool): Whether or not the state is terminal.
        """
        pass

    @abstractmethod
    def reset(self) -> State:
        """Reset the environment

        Returns:
            state (State): Returns the state of the new environment.
        """
        pass

    @abstractmethod
    def calculate_metrics(self) -> Metrics:
        """Calculate the metrics of the episode that just terminated.

        Returns:
            metrics (Metrics): The metrics at the end of the episode.
        """
        pass

    @abstractmethod
    def summarize_metrics_seq(
            self, metrics_seq: Sequence[Metrics]) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_random_policy(self) -> PolicyBase:
        """Get the random policy for the current enviornment.

        Returns:
            policy (PolicyBase): The random policy.
        """
        pass

    @abstractmethod
    def new_env_like(self) -> "Environment":
        """Create a new environment that has same configuration as this instance.
        """
        pass


class DiscreteEnvironment(Environment):

    ActionType = DiscreteAction

    @abstractmethod
    @overrides
    def step(self,
             action: Action,
             to_visualize: bool = False) -> Tuple[Action, State, float, bool]:
        pass
