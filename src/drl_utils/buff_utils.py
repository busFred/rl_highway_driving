from typing import List, Tuple

import numpy as np
import torch
from mdps.mdp_abc import Action, Environment, PolicyBase, State
from torch.utils.data import IterableDataset


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

    def add_experience(self, state: State, action: Action, next_state: State,
                       next_reward: float, is_terminal: bool):
        """Add an experience to the replay buffer

        Args:
            state (State): The current state.
            action (Action): The action taken at current time step.
            next_state (State): The next state given the current state and action.
            reward (float): The immediate reward associated with next_state.
            is_terminal (bool): Whether the next_state is a terminal state.
        """
        # remove sample from replay buffer if max_size is exceeded
        if len(self._replay_buff) >= self._max_size:
            self._replay_buff.pop(0)
        # add sample to replay buffer
        self._replay_buff.append(
            (state, action, next_reward, next_state, is_terminal))

    def sample_experiences(self,
                           batch_size: int,
                           dtype: torch.dtype = torch.float,
                           device: torch.device = torch.device("cpu")):
        """Sample a batch of expereince

        Args:
            batch_size (int): The size of the batch.
            dtype (torch.dtype, optional): The data type of states, rewards, and next_states. Defaults to torch.float.

        Returns:
            states (torch.Tensor): (batch_size, state_dim) The current states.
            actions (torch.Tensor): (batch_size, action_dim) The current actions.
            next_states (torch.Tensor): (batch_szie, state_dim) The next states given current states and actions.
            rewards (torch.Tensor): (batch_size, 1) The next immediate rewards.
            is_termianls (torch.Tensor): (batch_size, ) Whether next states are terminal.
        """
        # batch_size never exceeds maximum experiences in the buffer
        batch_size = min(len(self._replay_buff), batch_size)
        # get the sample to be included in the current batch
        batch_idxs = np.random.randint(len(self._replay_buff), size=batch_size)
        replay_buff_batch = [self._replay_buff[idx] for idx in batch_idxs]
        # convert to pytorch tensor
        # (batch_size, n_state_features)
        states: torch.Tensor = torch.tensor(
            np.asarray(
                [x[0].get_np_state(copy=True) for x in replay_buff_batch]))
        # (batch_size, n_action_features)
        actions: torch.Tensor = torch.tensor(
            np.asarray(
                [x[1].get_np_action(copy=True) for x in replay_buff_batch]))
        # (batch_size, 1)
        next_rewards: torch.Tensor = torch.tensor(
            np.asarray([x[2] for x in replay_buff_batch]))
        next_rewards = next_rewards.reshape(batch_size, 1)
        # (batch_size, n_state_features)
        next_states: torch.Tensor = torch.tensor(
            np.asarray(
                [x[3].get_np_state(copy=True) for x in replay_buff_batch]))
        # (batch_size, )
        is_terminals: torch.Tensor = torch.tensor(
            np.asarray([x[4] for x in replay_buff_batch]))
        # is_terminals = is_terminals.expand(batch_size, 1)
        # convert datatype
        states = states.to(dtype=dtype, device=device)
        actions = actions.to(device=device)
        next_rewards = next_rewards.to(dtype=dtype, device=device)
        next_states = next_states.to(dtype=dtype, device=device)
        is_terminals = is_terminals.to(device=device)
        return states, actions, next_states, next_rewards, is_terminals

    def __len__(self):
        return len(self._replay_buff)


class RLDataset(ReplayBuffer, IterableDataset):

    batch_size: int
    _dtype: torch.dtype
    _device: torch.device

    def __init__(self,
                 max_size: int,
                 batch_size: int,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device("cpu")):
        super().__init__(max_size=max_size)
        self.batch_size = batch_size
        self._dtype = dtype
        self._device = device

    def __iter__(self):
        return zip(*(self.sample_experiences(self.batch_size, self._dtype,
                                             self._device)))


def populate_replay_buffer(buff: ReplayBuffer, env: Environment,
                           policy: PolicyBase, target_size: int):
    """Create and initialize the replay buffer with random policy.

    Args:
        buff (ReplayBuffer): The buffer to be populated.
        env (Environment): The enviornment in use.
        policy (PolicyBase): The policy used to populate the buffer.
        target_size (int): The target replay buffer size.
    """
    state: State = env.reset()
    while len(buff) < min(target_size, buff._max_size):
        action: Action = policy.sample_action(state)
        next_state, action, next_reward, is_terminal = env.step(
            action=action, to_visualize=False)
        buff.add_experience(state=state,
                            action=action,
                            next_state=next_state,
                            next_reward=next_reward,
                            is_terminal=is_terminal)
        if is_terminal == False:
            state = next_state
        else:
            state = env.reset()
