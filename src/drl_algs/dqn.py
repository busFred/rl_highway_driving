import copy
from dataclasses import field
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from dataclasses_json import dataclass_json
from drl_utils.buff_utils import ReplayBuffer
from mdps import mdp_utils
from mdps.mdp_abc import DiscreteAction, DiscreteEnvironment, State
from torch import nn


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
    _optimizers: Union[Sequence[torch.optim.Optimizer], None]
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
    def optimizers(self) -> Sequence[torch.optim.Optimizer]:
        if self._optimizers is None:
            raise ValueError("self._optimizers is None")
        return self._optimizers

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
        optimizers: Sequence[torch.optim.Optimizer],
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self._dqns = dqns
        self._optimizers = optimizers
        self._targ_dqns = None
        self.dtype = dtype
        self.device = device
        self.eval()

    def predict_q_vals(self, states: Sequence[State]) -> torch.Tensor:
        """Predict and select the minimum q_vals predictions made by different dqns.

        Args:
            states (Sequence[State]): (n_states, ) States to be predicted.

        Returns:
            pred_q_vals (torch.Tensor): (n_states, n_actions) The predicted q values.
        """
        states_tensor: torch.Tensor = mdp_utils.states_to_torch(
            states=states, dtype=self.dtype, device=self.device)
        return self._predict_q_vals(states=states_tensor)

    def predict_q_vals_raw(self,
                           states: Sequence[State]) -> Sequence[torch.Tensor]:
        """Return the predictions of the Q Vales for all states 

        Args:
            states (Sequence[State]): (n_states, ) States to be predicted.

        Returns:
            pred_q_vals (torch.Tensor): (n_dqns, n_states, n_actions) The predicted q values.
        """
        states_tensor: torch.Tensor = mdp_utils.states_to_torch(
            states=states, dtype=self.dtype, device=self.device)
        return self._predict_q_vals_raw(states=states_tensor)

    def train(self):
        """Set the dqns to train mode.

        If self.targ_dqns is None, then make a deep copy of self.dqns and call self._update_targ()

        Raises:
            ValueError: self._optimizers has no optimizer registered
        """
        if self._optimizers is None:
            raise ValueError("self._optimizers has no optimizer registered")
        if self._targ_dqns is None:
            self._update_targ()
        for dqn, optimizer in zip(self.dqns, self.optimizers):
            dqn.train()
            optimizer.zero_grad()

    def eval(self):
        """Set the dqns to eval mode.
        """
        for dqn in self.dqns:
            dqn.eval()

    def _predict_q_vals(self, states: torch.Tensor) -> torch.Tensor:
        """Predict and select the minimum q_vals predictions made by different dqns.

        Args:
            states (torch.Tensor): (n_states, n_state_features) States to be predicted.

        Returns:
            q_vals_pred (torch.Tensor): (n_states, n_actions) The predicted q values.
        """
        q_vals_pred_s: Sequence[torch.Tensor] = self._predict_q_vals_raw(
            states=states)
        # (n_dqns, n_states, n_actions)
        q_vals_pred: torch.Tensor = torch.as_tensor(q_vals_pred_s,
                                                    dtype=self.dtype,
                                                    device=self.device)
        # (n_states, n_actions)
        q_vals_pred = q_vals_pred.min(0)[0]
        return q_vals_pred

    def _predict_q_vals_raw(self,
                            states: torch.Tensor) -> Sequence[torch.Tensor]:
        """Predict Q Vales from the Tensor representation of states.

        Args:
            states (torch.Tensor): (n_states, n_state_features) States to be predicted.

        Returns:
            q_vals_pred (torch.Tensor): (n_dqns, n_states, n_actions) The predicted q values.
        """
        states.to(dtype=self.dtype, device=self.device)
        # (n_dqns, n_states, n_actions)
        q_vals_pred_l: List[torch.Tensor] = [dqn(states) for dqn in self.dqns]
        return q_vals_pred_l

    def _compute_target(self, next_states: torch.Tensor,
                        next_rewards: torch.Tensor, is_terminals: torch.Tensor,
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
        next_states.to(dtype=self.dtype, device=self.device)
        target_q_vals_l: List[torch.Tensor] = list()
        # get target q_vals from each targ_dqns
        for targ_dqn in self.targ_dqns:
            # (n_states, n_actions)
            next_q_vals: torch.Tensor = targ_dqn(next_states)
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


def train_dqn(env: DiscreteEnvironment, dqn: DQN, dqn_config: DQNConfig):
    dqn.train()
    buff: ReplayBuffer = ReplayBuffer.create_random_replay_buffer(
        env,
        max_size=dqn_config.max_buff_size,
        target_size=dqn_config.max_buff_size)
    for curr_eps in range(dqn_config.n_episodes):
        state: State = env.reset()
        for cur_step in range(dqn_config.max_episode_steps):
            next_state: State = _deep_q_step(env=env,
                                             state=state,
                                             dqn=dqn,
                                             dqn_config=dqn_config,
                                             buff=buff)
            state = next_state
        if curr_eps % dqn_config.targ_update_episodes == 0:
            dqn._update_targ()
        print(str.format("episode: {}/{}", curr_eps + 1,
                         dqn_config.n_episodes))


def _deep_q_step(env: DiscreteEnvironment, state: State, dqn: DQN,
                 dqn_config: DQNConfig, buff: ReplayBuffer) -> State:
    """Step for deep q network

    Args:
        env (DiscreteEnvironment): A discrete enviornment.
        state (State): The current state of the enviornment.
        dqn (DQN): The dqn.
        dqn_config (DQNConfig): The hyperparameters for the dqn.
        buff (ReplayBuffer): The replay buffer.

    Returns:
        State: The next state.
    """
    dqn.eval()
    action, next_state, next_reward, is_terminal = _eps_greedy_step(
        env=env, state=state, dqn=dqn, dqn_config=dqn_config)
    buff.add_experience(state=state,
                        action=action,
                        next_state=next_state,
                        next_reward=next_reward,
                        is_terminal=is_terminal)
    states, actions, next_states, next_rewards, is_terminals = buff.sample_experiences(
        dqn_config.batch_size, dqn.dtype, dqn.device)
    dqn.train()
    target_q_vals: torch.Tensor = dqn._compute_target(
        next_states=next_states,
        next_rewards=next_rewards,
        is_terminals=is_terminals,
        dqn_config=dqn_config)
    pred_q_vals_s: Sequence[torch.Tensor] = dqn._predict_q_vals_raw(states)
    for pred_q_vals, optimizer in zip(pred_q_vals_s, dqn.optimizers):
        error: torch.Tensor = nn.SmoothL1Loss()(pred_q_vals, target_q_vals)
        error.backward()
        optimizer.step()
    return next_state


def _eps_greedy_step(
        env: DiscreteEnvironment, state: State, dqn: DQN,
        dqn_config: DQNConfig) -> Tuple[DiscreteAction, State, float, bool]:
    """Take a step based on epsilon greedy policy.

    Args:
        env (DiscreteEnvironment): A discrete enviornment.
        state (State): The current state of the enviornment.
        dqn (DQN): The dqn.
        dqn_config (DQNConfig): The hyperparameters for the dqn.

    Returns:
        action (DiscreteAction): The current action.
        next_state (State): The next state.
        next_reward (float): The next reward.
        is_terminal (bool): Whether next state is terminal.
    """
    is_random: bool = np.random.uniform(0, 1) < dqn_config.epsilon
    if is_random:
        action, next_state, next_reward, is_terminal = env.step_random()
        return action, next_state, next_reward, is_terminal
    next_q_vals: torch.Tensor = dqn.predict_q_vals(states=[state])
    action: DiscreteAction = env.int_to_action(next_q_vals.argmax(1)[0].item())
    next_state, next_reward, is_terminal = env.step(action)
    return action, next_state, next_reward, is_terminal
