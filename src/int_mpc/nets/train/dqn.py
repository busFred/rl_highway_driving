import copy
from dataclasses import dataclass, field
from typing import List, Sequence, Union

import numpy as np
import torch
from dataclasses_json import dataclass_json
from torch import nn

from ...mdps import mdp_utils
from ...mdps.mdp_abc import Environment, State
from ...utils import replay_buff_utils
from ...utils.replay_buff_utils import ReplayBuffer


@dataclass_json
class DQNConfig:
    epsilon: float = field(default=0.5)
    discount: float = field(default=1.0)
    n_episodes: int = field(default=1000)
    max_episode_steps: int = field(default=30)
    max_buff_size: int = field(default=10000)
    batch_size: int = field(default=100)
    targ_update_episodes: int = field(default=20)


class DQN:

    _dqns: Sequence[nn.Module]
    _targ_dqns: Union[Sequence[nn.Module], None]
    dtype: torch.dtype
    device: torch.device

    @property
    def dqns(self) -> Sequence[nn.Module]:
        """Get the dqns.

        Returns:
            dqns (Sequence[nn.Module]): The self.dqns.
        """
        for dqn in self._dqns:
            dqn.to(device=self.device, dtype=self.dtype)
        return self.dqns

    @property
    def targ_dqns(self) -> Sequence[nn.Module]:
        """Get the targ_dqns.

        Raises:
            ValueError: if self.targ_dqns is None.

        Returns:
            targ_dqns (Sequence[nn.Module]): The target self.dqns.
        """
        if self._targ_dqns is None:
            raise ValueError("self.targ_dqns is None")
        for targ_dqn in self._targ_dqns:
            targ_dqn.to(device=self.device, dtype=self.dtype)
        return self._targ_dqns

    def __init__(
        self,
        dqns: Sequence[nn.Module],
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self._dqns = dqns
        self._targ_dqns = None
        self.dtype = dtype
        self.device = device
        self.eval()

    def predict_q_vals(self, states: Sequence[State]) -> torch.Tensor:
        """Predict Q Vales of a sequence of states.

        Args:
            states (Sequence[State]): (n_batch_size, ) States to be predicted.

        Returns:
            pred_q_vals (torch.Tensor): (n_batch_size, n_dqns) The predicted q values.
        """
        states_tensor: torch.Tensor = mdp_utils.states_to_torch(
            states=states, dtype=self.dtype, device=self.device)
        return self._predict_q_vals(states=states_tensor)

    def train(self):
        """Set the dqns to train mode.

        Raises:
            ValueError: self.targ_dqns is none.
        """
        if self._targ_dqns is None:
            self._update_targ()
        for dqn in self.dqns:
            dqn.train()

    def eval(self):
        """Set the dqns to eval mode.
        """
        for dqn in self.dqns:
            dqn.eval()

    def _predict_q_vals(self, states: torch.Tensor) -> torch.Tensor:
        """Predict Q Vales from the Tensor representation of states.

        Args:
            states (torch.Tensor): (n_states, n_state_features) States to be predicted.

        Returns:
            q_vals_pred (torch.Tensor): (n_dqns, n_states, n_actions) The predicted q values.
        """
        states.to(dtype=self.dtype, device=self.device)
        # (n_dqns, n_states, n_actions)
        q_vals_pred_l: Sequence[torch.Tensor] = [
            dqn(states) for dqn in self.dqns
        ]
        q_vals_pred: torch.Tensor = torch.as_tensor(q_vals_pred_l,
                                                    dtype=self.dtype,
                                                    device=self.device)
        return q_vals_pred

    def _compute_target(self, states: torch.Tensor, next_rewards: torch.Tensor,
                        is_terminals: torch.Tensor,
                        dqn_config: DQNConfig) -> torch.Tensor:
        """Compute target q values given a tensor representation for states.

        Args:
            states (torch.Tensor): (n_states, n_state_features)
            next_rewards (torch.Tensor): (n_states, 1)
            is_terminals (torch.Tensor): (n_states, )
            dqn_config (DQNConfig): dqn hyperparameter

        Returns:
            target_q_vals (torch.Tensor): (n_states, 1) The target value.
        """
        states.to(dtype=self.dtype, device=self.device)
        target_q_vals_l: List[torch.Tensor] = list()
        # get target q_vals from each targ_dqns
        for targ_dqn in self.targ_dqns:
            # (n_states, n_actions)
            next_q_vals: torch.Tensor = targ_dqn(states)
            next_q_vals = next_q_vals.detach()
            # select next action greedily
            # (n_states, 1)
            next_q_vals = next_q_vals.max(1)[0]
            # current terminal states has no future reward
            next_q_vals[is_terminals] = 0.0
            target_q_vals: torch.Tensor = next_rewards + dqn_config.discount * next_q_vals
            target_q_vals_l.append(target_q_vals)
        # use conservative estimation
        # (n_dqns, n_states, 1)
        target_q_vals: torch.Tensor = torch.as_tensor(target_q_vals_l,
                                                      dtype=self.dtype,
                                                      device=self.device)
        # (n_states, 1)
        target_q_vals = target_q_vals.min(0)[0]
        return target_q_vals

    def _update_targ(self):
        """Update target network.
        """
        if self._targ_dqns is None:
            self._targ_dqns = copy.deepcopy(self.dqns)
        for dqn, targ_dqn in zip(self.dqns, self.targ_dqns):
            targ_dqn.load_state_dict(dqn.state_dict())
            targ_dqn.eval()


def train_dqn(env: Environment, dqn: DQN, dqn_config: DQNConfig):
    dqn.train()
    buff: ReplayBuffer = ReplayBuffer.create_random_replay_buffer(
        env,
        max_size=dqn_config.max_buff_size,
        target_size=dqn_config.max_buff_size)
