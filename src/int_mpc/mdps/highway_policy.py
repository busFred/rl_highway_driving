from typing import List, Sequence

import numpy as np
import scipy.stats
import torch
from mdps.policy_abc import StochActNetPolicyBase
from overrides.overrides import overrides
from torch import nn

from .highway_mdp import HighwayEnvDiscreteAction, HighwayEnvState


class StochDiscActNet(StochActNetPolicyBase):
    def __init__(
        self,
        action_net: nn.Module,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__(action_net=action_net, dtype=dtype, device=device)

    @overrides
    def sample_action(
        self, states: Sequence[HighwayEnvState]
    ) -> Sequence[HighwayEnvDiscreteAction]:
        pvals_batch: np.ndarray = self.eval_action_net(states=states)
        actions: List[HighwayEnvDiscreteAction] = list()
        for p_vals in pvals_batch:
            action_i: int = np.argmax(
                scipy.stats.multinomial.rvs(n=1, p=p_vals))[0]
            actions.append(HighwayEnvDiscreteAction(action_i))
        return actions

    @overrides
    def eval_entropy(self, states: Sequence[HighwayEnvState]) -> np.ndarray:
        pvals_batch: np.ndarray = self.eval_action_net(states=states)
        entropy_outputs: np.ndarray = scipy.stats.multinomial.entropy(
            n=1, p=pvals_batch)
        return entropy_outputs
