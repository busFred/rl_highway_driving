from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Tuple

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


class DiscreteAction(Action, IntEnum):
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


class Environment(ABC):
    @abstractmethod
    def step(self, action: Action) -> Tuple[State, float, bool]:
        """Take an action.

        Args:
            action (Action): The action to be taken.

        Returns:
            state (State): The next state after taking the passed in action.
            reward (float): The reward associated with the state.
            is_terminal (bool): Whether or not the state is terminal.
        """
        pass

    @abstractmethod
    def step_random(self) -> Tuple[Action, State, float, bool]:
        """Take a random action.

        Returns:
            action (Action): The random action being taken.
            next_state (State): The next state after taking the passed in action.
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


class DiscreteEnvironment(Environment):
    @overrides
    @abstractmethod
    def step(self, action: DiscreteAction) -> Tuple[State, float, bool]:
        pass

    @overrides
    @abstractmethod
    def step_random(self) -> Tuple[DiscreteAction, State, float, bool]:
        """Take a random action.

        Returns:
            action (Action): The random action being taken.
            next_state (State): The next state after taking the passed in action.
            reward (float): The reward associated with the state.
            is_terminal (bool): Whether or not the state is terminal.
        """
        pass

    @abstractmethod
    def int_to_action(self, action: int) -> "DiscreteAction":
        pass
