import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from dataclasses_json import dataclass_json
from drl_utils.buff_utils import ReplayBuffer
from mdps import mdp_utils
from mdps.mdp_abc import DiscreteAction, DiscreteEnvironment, State
from torch import nn


@dataclass_json
@dataclass
class DQNConfig:
    epsilon: float = field(default=0.5)
    epsilon_decay: float = field(default=0.99)
    epsilon_update_episodes: int = field(default=20)
    discount: float = field(default=1.0)
    n_episodes: int = field(default=1000)
    max_episode_steps: int = field(default=30)
    max_buff_size: int = field(default=10000)
    batch_size: int = field(default=100)
    targ_update_episodes: int = field(default=20)


class DQN:

    _dqn: nn.Module
    _optimizer: Union[torch.optim.Optimizer, None]
    _targ_dqn: Union[nn.Module, None]
    dtype: torch.dtype
    device: torch.device

    @property
    def dqn(self) -> nn.Module:
        """Get the dqn.

        Returns:
            dqn (nn.Module): The self.dqn.
        """
        self._dqn.to(device=self.device, dtype=self.dtype)
        return self._dqn

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            raise ValueError("self._optimizer is None")
        return self._optimizer

    @property
    def targ_dqn(self) -> nn.Module:
        """Get the targ_dqn.

        Raises:
            ValueError: if self.targ_dqn is None.

        Returns:
            targ_dqn (nn.Module): The target self.dqn.
        """
        if self._targ_dqn is None:
            raise ValueError("self.targ_dqn is None")
        self._targ_dqn.to(device=self.device, dtype=self.dtype)
        return self._targ_dqn

    def __init__(
        self,
        dqn: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self._dqn = dqn
        self._optimizer = optimizer
        self._targ_dqn = None
        self.dtype = dtype
        self.device = device
        self.eval()

    def predict_q_vals(self, states: Sequence[State]) -> torch.Tensor:
        """Predict and select the q_vals predictions made by self.dqn.

        Args:
            states (Sequence[State]): (n_states, ) States to be predicted.

        Returns:
            q_vals (torch.Tensor): (n_states, n_actions) The predicted q values.
        """
        states_tensor: torch.Tensor = mdp_utils.states_to_torch(
            states=states, dtype=self.dtype, device=self.device)
        return self._predict_q_vals(states=states_tensor)

    def _predict_q_vals(self, states: torch.Tensor) -> torch.Tensor:
        """Predict and select the q_vals predictions made by self.dqn.

        Args:
            states (torch.Tensor): (n_states, n_state_features) States to be predicted.

        Returns:
            q_vals (torch.Tensor): (n_states, n_actions) The predicted q values.
        """
        states = states.to(device=self.device, dtype=self.dtype)
        # (n_states, n_actions)
        q_vals: torch.Tensor = self.dqn(states)
        return q_vals

    def train(self):
        """Set the dqn to train mode.

        If self.targ_dqn is None, then make a deep copy of self.dqn and call self._update_targ()
        """
        if self._targ_dqn is None:
            self._update_targ()
        self.dqn.train()

    def eval(self):
        """Set the dqn to eval mode.
        """
        self.dqn.eval()

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
        # (n_states, n_actions)
        next_q_vals: torch.Tensor = self.targ_dqn(next_states)
        next_q_vals = next_q_vals.detach()
        # select next action greedily
        # (n_states, 1)
        next_q_vals = next_q_vals.max(1, keepdim=True)[0]
        # current terminal states has no future reward
        next_q_vals[is_terminals] = 0.0
        # (n_states, 1)
        target_q_vals: torch.Tensor = next_rewards + dqn_config.discount * next_q_vals
        return target_q_vals

    def _update_targ(self):
        """Update target network.
        """
        if self._targ_dqn is None:
            self._targ_dqn = copy.deepcopy(self._dqn)
        self.targ_dqn.load_state_dict(self.dqn.state_dict())
        self.targ_dqn.eval()


def train_dqn(env: DiscreteEnvironment,
              dqn: DQN,
              dqn_config: DQNConfig,
              replay_buffer: Optional[ReplayBuffer] = None) -> ReplayBuffer:
    """Train the passed in DQN.

    Args:
        env (DiscreteEnvironment): The environment to be learned.
        dqn (DQN): The dqn to be trained. Modified inplace.
        dqn_config (DQNConfig): The hyperparameter for the DQN.
        replay_buffer (Optional[ReplayBuffer], optional): The replay buffer. Generated with random policy if None is provided. Defaults to None.

    Returns:
        ReplayBuffer: The current replay buffer.
    """
    # do not directly modify input configuration
    config: DQNConfig = copy.deepcopy(dqn_config)
    dqn.train()
    if replay_buffer is None:
        replay_buffer = ReplayBuffer.create_random_replay_buffer(
            env, max_size=config.max_buff_size, target_size=config.batch_size)
    for curr_eps in range(config.n_episodes):
        state: State = env.reset()
        for cur_step in range(config.max_episode_steps):
            next_state, is_terminal = _deep_q_step(env=env,
                                                   state=state,
                                                   dqn=dqn,
                                                   dqn_config=config,
                                                   buff=replay_buffer)
            if is_terminal:
                break
            state = next_state
        if curr_eps % config.epsilon_update_episodes == 0:
            config.epsilon = config.epsilon * config.epsilon_decay
        if curr_eps % config.targ_update_episodes == 0:
            dqn._update_targ()
        print(str.format("episode: {}/{}", curr_eps + 1, config.n_episodes))
    return replay_buffer


def greedy_step(
        env: DiscreteEnvironment,
        state: State,
        dqn: DQN,
        to_vis: bool = False) -> Tuple[DiscreteAction, State, float, bool]:
    """Take a step based on epsilon greedy policy.

    Args:
        env (DiscreteEnvironment): A discrete enviornment.
        state (State): The current state of the enviornment.
        dqn (DQN): The dqn.
        to_vis (bool): Whether to visualize or not.

    Returns:
        action (DiscreteAction): The current action.
        next_state (State): The next state.
        next_reward (float): The next reward.
        is_terminal (bool): Whether next state is terminal.
    """
    # (1, n_actions)
    next_q_vals: torch.Tensor = dqn.predict_q_vals(states=[state])
    action: DiscreteAction = env.int_to_action(next_q_vals.argmax(1)[0].item())
    next_state, next_reward, is_terminal = env.step(action,
                                                    to_visualize=to_vis)
    return action, next_state, next_reward, is_terminal


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
        action, next_state, next_reward, is_terminal = env.step_random(
            to_visualize=False)
        return action, next_state, next_reward, is_terminal
    # (1, n_actions)
    next_q_vals: torch.Tensor = dqn.predict_q_vals(states=[state])
    action: DiscreteAction = env.int_to_action(next_q_vals.argmax(1)[0].item())
    next_state, next_reward, is_terminal = env.step(action, to_visualize=False)
    return action, next_state, next_reward, is_terminal


def _deep_q_step(env: DiscreteEnvironment, state: State, dqn: DQN,
                 dqn_config: DQNConfig,
                 buff: ReplayBuffer) -> Tuple[State, bool]:
    """Step for deep q network

    Args:
        env (DiscreteEnvironment): A discrete enviornment.
        state (State): The current state of the enviornment.
        dqn (DQN): The dqn.
        dqn_config (DQNConfig): The hyperparameters for the dqn.
        buff (ReplayBuffer): The replay buffer.

    Returns:
        next_state (State): The next state.
        is_termianl (bool): Whether next state is terminal.
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
    dqn.optimizer.zero_grad()
    target_q_vals: torch.Tensor = dqn._compute_target(
        next_states=next_states,
        next_rewards=next_rewards,
        is_terminals=is_terminals,
        dqn_config=dqn_config)
    # (n_states, n_actions)
    pred_q_vals: torch.Tensor = dqn._predict_q_vals(states=states)
    # (n_states, 1)
    pred_q_vals = pred_q_vals.gather(1, actions)
    error: torch.Tensor = nn.SmoothL1Loss()(pred_q_vals, target_q_vals)
    error.backward()
    dqn.optimizer.step()
    return next_state, is_terminal
