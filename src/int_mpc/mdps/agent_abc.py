from abc import ABC, abstractmethod
import numpy as np

from torch import nn
import torch

from .mdp_abc import Action, State


class Policy(ABC):
    @abstractmethod
    def sample_action(self, state: State) -> Action:
        """Given a state, return an action

        Args:
            state (State): The current state.

        Returns:
            Action: The current action.
        """
        pass


class ActionNetPolicy(Policy):

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

    def _eval_action_net(
        self,
        state: State,
    ):
        """Evaluate action net.

        Args:
            state (State): The current state.

        Returns:
            policy_param (np.ndarray): The parameter of the distribution of the policy.
        """
        # set action net to eval mode
        self.action_net.eval()
        # set to correct data type and move to correct device
        self.action_net.to(device=self.device, dtype=self.dtype)
        # convert State to torch.Tensor
        state_np: np.ndarray = state.get_np_state()
        state_torch = torch.tensor(state_np,
                                   dtype=self.dtype,
                                   device=self.device)
        # forward pass action net to get action distribution
        policy_param_torch: torch.Tensor = self.action_net(state_torch)
        policy_param: np.ndarray = policy_param_torch.cpu().detach().numpy()
        return policy_param
