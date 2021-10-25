import copy
from typing import List, Sequence

import numpy as np
import torch
from torch import nn

from ...mdps.agent_abc import ActionNetPolicy
from ...mdps.mdp_abc import Action, DiscreteAction, Environment, State
from ...utils import replay_buffer_utils
from ...utils.replay_buffer_utils import ReplayBuffer


def train_discrete_sac(
    env: Environment,
    q_nets: Sequence[nn.Module],
    policy: ActionNetPolicy,
    discount: float,
    num_episodes: int,
    max_episode_steps: int,
    batch_size: int,
    max_buffer_size: int,
    targ_update_episodes: int,
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    # at least one and at most two q_nets
    if len(q_nets) < 1 or len(q_nets) > 2:
        raise ValueError
    # make target_q_nets by cloning the input q_nets
    target_q_nets: Sequence[nn.Module] = [
        copy.deepcopy(q_net) for q_net in q_nets
    ]
    replay_buffer: ReplayBuffer = replay_buffer_utils.create_random_replay_buffer(
        env=env,
        max_size=max_buffer_size,
        target_size=int(0.8 * max_buffer_size))
    for curr_episode in range(num_episodes):
        # make sure target is updated at least once for each episode
        _update_target_q_nets(q_nets=q_nets, target_q_nets=target_q_nets)
        # create a new environment
        state: State = env.reset()
        for curr_step in range(max_episode_steps):
            # next_state: State = _sac_step()
            pass


def _discrete_sac_step(
    env: Environment,
    state: State,
    q_nets: Sequence[nn.Module],
    target_q_nets: Sequence[nn.Module],
    policy: ActionNetPolicy,
    replay_buffer: ReplayBuffer,
    discount: float,
    num_episodes: int,
    max_episode_steps: int,
    batch_size: int,
    max_buffer_size: int,
    targ_update_episodes: int,
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu")
) -> State:
    action: Action = policy.sample_action(state=state)
    next_state, next_reward, is_terminal = env.step(action=action)
    replay_buffer.add_experience(state=state,
                                 action=action,
                                 next_state=next_state,
                                 next_reward=next_reward,
                                 is_terminal=is_terminal)
    states, actions, next_states, next_rewards, is_terminals = replay_buffer.sample_expereinces(
        batch_size=batch_size, dtype=dtype)
    target_q_vals: torch.Tensor = _compute_target_q_vals(
        next_states=next_states,
        next_rewards=next_rewards,
        is_terminals=is_terminals,
        target_q_nets=target_q_nets,
        discount=discount)


def _update_target_q_nets(q_nets: Sequence[nn.Module],
                          target_q_nets: Sequence[nn.Module]):
    """Clone the q_nets to be used for computing target.

    Args:
        q_nets (Sequence[nn.Module]): The q_nets to be cloned.
    """
    for q_net, target_q_net in zip(q_nets, target_q_nets):
        target_q_net.load_state_dict(q_net.state_dict())
        target_q_net.eval()


def _compute_target_q_vals(next_states: torch.Tensor,
                           next_rewards: torch.Tensor,
                           is_terminals: torch.Tensor,
                           target_q_nets: Sequence[nn.Module],
                           discount: float) -> torch.Tensor:
    """Compute the target q_nets

    Args:
        next_states (torch.Tensor): The next states.
        next_rewards (torch.Tensor): The next rewards.
        target_q_nets (Sequence[nn.Module]): The q_nets used to evaluate target. Must be set to eval ahead of time.
        discount (float): The discount.

    Returns:
        targets (torch.Tensor): The `targets = next_rewards + discount * next_q_vals`.
    """
    next_q_vals: torch.Tensor = _compute_min_q_vals(next_states,
                                                    q_nets=target_q_nets)
    next_q_vals = next_q_vals.detach()
    next_q_vals[is_terminals] = 0.0
    targets: torch.Tensor = next_rewards + discount * next_rewards
    return targets


def _compute_q_vals(states: torch.Tensor,
                    q_nets: Sequence[nn.Module]) -> torch.Tensor:
    """Compute the Q values with q_nets.

    The function does NOT modify the mode that the network is in. The q_net.eval or q_net.train is not called.

    Args:
        states (torch.Tensor): The states to be evaluated with the q_nets.
        q_nets (Sequence[nn.Module]): The q_nets to be used to evaluate.

    Returns:
        q_vals (torch.Tensor): (n_states, n_q_nets) The q values evaluated by different q_net.
    """
    q_vals_l: List[torch.Tensor] = list()
    for q_net in q_nets:
        # TODO confirm shape
        # q_val SHOULD have shape (n_states, 1)
        q_val: torch.Tensor = q_net(states)
        q_val = q_val.detach()
        q_vals_l.append(q_val)
    q_vals = torch.cat(q_vals_l, dim=1)
    return q_vals


def _compute_min_q_vals(states: torch.Tensor,
                        q_nets: Sequence[nn.Module]) -> torch.Tensor:
    """Compute minimum Q values.

    The function does NOT modify the mode that the network is in. The q_net.eval or q_net.train is not called.

    Args:
        states (torch.Tensor): The states to be evaluated with the q_nets.
        q_nets (Sequence[nn.Module]): The q_nets to be used to evaluate.

    Returns:
        min_q_vals (torch.Tensor): (n_states, 1) The minimum q values among the two.
    """
    q_vals: torch.Tensor = _compute_q_vals(states, q_nets)
    min_q_vals: torch.Tensor = torch.min(q_vals, dim=1, keepdim=True).values
    return min_q_vals