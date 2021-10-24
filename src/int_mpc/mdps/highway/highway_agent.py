import numpy as np
import torch
from overrides.overrides import overrides
from torch import nn

from ..agent_abc import ActionNetPolicy
from .highway_mdp import HighwayEnvDiscreteAction, HighwayEnvState


class HighwayAgentNetPolicy(ActionNetPolicy):
    rng: np.random.Generator

    def __init__(
        self,
        action_net: nn.Module,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__(action_net=action_net, dtype=dtype, device=device)
        self.rng = np.random.default_rng()

    @overrides
    def sample_action(self,
                      state: HighwayEnvState) -> HighwayEnvDiscreteAction:
        """Sample an action from the policy.

        Args:
            state (HighwayEnvState): The current state.

        Returns:
            action (HighwayEnvDiscreteAction): The current action given current state.
        """
        pvals: np.ndarray = self._eval_action_net(state=state)
        action_i: int = np.argmax(self.rng.multinomial(n=1, pvals=pvals))[0]
        action = HighwayEnvDiscreteAction(action_i)
        return action
