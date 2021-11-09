from abc import ABC, abstractmethod
from typing import Sequence

from .mdp_abc import Action, State


class PolicyBase(ABC):
    @abstractmethod
    def sample_action(self, state: State) -> Action:
        pass
