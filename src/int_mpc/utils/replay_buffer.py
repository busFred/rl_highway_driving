from typing import List, Tuple

import numpy as np
import torch

from ..envs.env_abc import Action, State


class ReplayBuffer:
    _replay_buff: List[Tuple[State, Action, float, State, bool]]
    _max_size: int

    def __init__(self, max_size: int):
        """Constructor for ReplayBuffer

        Args:
            max_size (int): The maximum capacity of the replay buffer.
        """
        self._replay_buff = list()
        self._max_size = max_size

    def add_experience(self, state: State, action: Action, reward: float,
                       next_state: State, is_terminal: bool):
        """Add an experience to the replay buffer

        Args:
            state (State): The current state.
            action (Action): The action taken at current time step.
            reward (float): The immediate reward associated with next_state.
            next_state (State): The next state given the current state and action.
            is_terminal (bool): Whether the next_state is a terminal state.
        """
        # remove sample from replay buffer if max_size is exceeded
        if len(self._replay_buff) >= self._max_size:
            self._replay_buff.pop(0)
        # add sample to replay buffer
        self._replay_buff.append(
            (state, action, reward, next_state, is_terminal))

    def sample_expereinces(self,
                           batch_size: int,
                           dtype: torch.dtype = torch.float):
        """Sample a batch of expereince

        Args:
            batch_size (int): The size of the batch.
            dtype (torch.dtype, optional): The data type of states, rewards, and next_states. Defaults to torch.float.

        Returns:
            states (torch.Tensor): (batch_size, state_dim) The current states.
            actions (torch.Tensor): (batch_size, action_dim) The current actions.
            rewards (torch.Tensor): (batch_size, 1) The next immediate rewards.
            next_states (torch.Tensor): (batch_szie, state_dim) The next states given current states and actions.
            is_termianls (torch.Tensor): (batch_size, 1) Whether next states are terminal.
        """
        # batch_size never exceeds maximum experiences in the buffer
        batch_size = min(len(self._replay_buff), batch_size)
        # get the sample to be included in the current batch
        batch_idxs = np.random.randint(len(self._replay_buff), size=batch_size)
        replay_buff_batch = [self._replay_buff[idx] for idx in batch_idxs]
        # convert pytorch tensor
        states: torch.Tensor = torch.tensor(
            [x[0].get_np_state(copy=True) for x in replay_buff_batch])
        actions: torch.Tensor = torch.tensor(
            [x[1].get_np_action(copy=True) for x in replay_buff_batch])
        rewards: torch.Tensor = torch.tensor([x[2] for x in replay_buff_batch])
        rewards = rewards.expand(-1, 1)
        next_states: torch.Tensor = torch.tensor(
            [x[3].get_np_state(copy=True) for x in replay_buff_batch])
        is_terminals: torch.Tensor = torch.tensor(
            [x[4] for x in replay_buff_batch])
        is_terminals = is_terminals.expand(-1, 1)
        # convert datatype
        states = states.to(dtype=dtype)
        rewards = rewards.to(dtype=dtype)
        next_states = next_states.to(dtype=dtype)
        return states, actions, rewards, next_states, is_terminals

    def __len__(self):
        return len(self._replay_buff)
