from abc import ABC, abstractmethod
import collections
from typing import Sequence, Union
import numpy as np
from overrides.overrides import overrides

from torch import nn
import torch

from .mdp_abc import Action, State
from . import mdp_utils


class PolicyBase(ABC):
    @abstractmethod
    def sample_action(self, states: Sequence[State]) -> Sequence[Action]:
        """Given current states, return current actions.

        Args:
            states (State): (n_batch_size, ) The current states.

        Raises:
            ValueError: if states is torch.Tensor

        Returns:
            actions (Sequence[Action]): (n_batch_size) The current actions.
        """
        if isinstance(states, torch.Tensor):
            raise ValueError


class ActNetPolicyBase(PolicyBase):

    action_net: nn.Module
    dtype: torch.dtype
    device: torch.device

    def __init__(
        self,
        action_net: nn.Module,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:
        action_net.to(device=device, dtype=dtype)
        self.action_net = action_net
        self.dtype = dtype
        self.device = device

    @abstractmethod
    @overrides
    def sample_action(self, states: Sequence[State]) -> Sequence[Action]:
        """Given current states, return current actions.

        Subclass must call super().sample_action() to get conversion from Sequence[State] to torch.Tensor.

        Args:
            states (State): (n_batch_size, ) The current states.

        Returns:
            actions (Sequence[Action]): (n_batch_size) The current actions.
        """
        pass

    def eval_action_net(self, states: Sequence[State]) -> np.ndarray:
        """Evaluate action net.

        Args:
            states (Sequence[State] | torch.Tensor): (n_batch_size, ) The current states.

        Returns:
            action_outputs (np.ndarray): (n_batch_size, action_net_output_dims) The output of the self.action_net.
        """
        # convert states to torch.Tensor
        states_torch = mdp_utils.states_to_torch(states, self.dtype,
                                                 self.device)
        # set action net to eval mode
        self.action_net.eval()
        # set to correct data type and move to correct device
        self.action_net.to(device=self.device, dtype=self.dtype)
        # forward pass action net to get action distribution
        action_outputs_torch: torch.Tensor = self.action_net(states_torch)
        action_outputs: np.ndarray = action_outputs_torch.cpu().detach().numpy(
        )
        return action_outputs


class StochActNetPolicyBase(ActNetPolicyBase):
    @abstractmethod
    def eval_entropy(self, states: Sequence[State]) -> np.ndarray:
        """Evaluate the entropy of the policy distribution.

        Subclass must call super().sample_action() to get conversion from Sequence[State] to torch.Tensor.

        Args:
            states (Sequence[State]): (n_batch_size, ) The current states.

        Returns:
            entropy_outputs (np.ndarray): (n_batch_size, 1) The entropies of the policy distribution.
        """
        pass