from abc import ABC, abstractmethod
from typing import Tuple


class Action(ABC):
    pass


class State(ABC):
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
    def reset(self) -> State:
        """Reset the environment

        Returns:
            state (State): Returns the state of the new environment.
        """
        pass
