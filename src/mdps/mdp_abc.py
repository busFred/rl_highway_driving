from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, Tuple, TypeVar

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


StateTypeVar = TypeVar("StateTypeVar", bound=State)
ActionTypeVar = TypeVar("ActionTypeVar", bound=Action)


class PolicyBase(ABC, Generic[StateTypeVar, ActionTypeVar]):
    @abstractmethod
    def sample_action(self, state: StateTypeVar) -> ActionTypeVar:
        pass


StateTypeVar = TypeVar("StateTypeVar", bound=State)
ActionTypeVar = TypeVar("ActionTypeVar", bound=Action)
MetricsTypeVar = TypeVar("MetricsTypeVar", bound=Metrics)


class Environment(ABC, Generic[StateTypeVar, ActionTypeVar, MetricsTypeVar]):
    @abstractmethod
    def step(self,
             action: ActionTypeVar,
             to_visualize: bool = False) -> Tuple[StateTypeVar, float, bool]:
        """Take an action.

        Args:
            action (Action): The action to be taken.
            to_visualize (bool): Whether to render the visualization.

        Returns:
            state (State): The next state after taking the passed in action.
            reward (float): The reward associated with the state.
            is_terminal (bool): Whether or not the state is terminal.
        """
        pass

    @abstractmethod
    def reset(self) -> StateTypeVar:
        """Reset the environment

        Returns:
            state (State): Returns the state of the new environment.
        """
        pass

    @abstractmethod
    def calculate_metrics(self) -> MetricsTypeVar:
        """Calculate the metrics of the episode that just terminated.

        Returns:
            metrics (Metrics): The metrics at the end of the episode.
        """
        pass

    @abstractmethod
    def get_random_policy(self) -> PolicyBase[StateTypeVar, ActionTypeVar]:
        """Get the random policy for the current enviornment.

        Returns:
            policy (PolicyBase): The random policy.
        """
        pass


StateTypeVar = TypeVar("StateTypeVar", bound=State)
ActionTypeVar = TypeVar("ActionTypeVar", bound=DiscreteAction)
MetricsTypeVar = TypeVar("MetricsTypeVar", bound=Metrics)


class DiscreteEnvironment(Environment[StateTypeVar, ActionTypeVar,
                                      MetricsTypeVar]):
    @abstractmethod
    @overrides
    def step(self,
             action: ActionTypeVar,
             to_visualize: bool = False) -> Tuple[StateTypeVar, float, bool]:
        pass
    