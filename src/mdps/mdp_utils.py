from typing import Sequence

import numpy as np
import torch

from .mdp_abc import Action, Environment, Metrics, PolicyBase, State


def states_to_torch(states: Sequence[State], dtype: torch.dtype,
                    device: torch.device) -> torch.Tensor:
    states_np: np.ndarray = np.asarray(
        [state.get_np_state() for state in states])
    states_torch: torch.Tensor = torch.tensor(states_np,
                                              dtype=dtype,
                                              device=device)
    return states_torch


def simulate(env: Environment,
             policy: PolicyBase,
             max_episode_steps: int,
             to_visualize: bool = True) -> Metrics:
    """Simulate the MDP with the given policy.

    Args:
        env (Environment): The enviornment to be simulated.
        policy (PolicyBase): The policy used to generate actions.
        max_episode_steps (int): The maximum number of episodes.
        to_visualize (bool, optional): Whether to visualize the enviorinment. Defaults to True.

    Returns:
        metrics (Metrics): The metrics of the current episode.
    """
    state: State = env.reset()
    # step until timeout occurs
    for curr_step in range(max_episode_steps):
        action: Action = policy.sample_action(state)
        next_state, _, _, is_terminal = env.step(action=action,
                                                 to_visualize=to_visualize)
        state = next_state
        if is_terminal:
            break
    metrics = env.calculate_metrics()
    return metrics


def simulate_episodes(env: Environment,
                      policy: PolicyBase,
                      max_episode_steps: int,
                      n_episodes: int,
                      to_visualize: bool = True) -> Sequence[Metrics]:
    """Simulate the MDP with the given policy.

    Args:
        env (Environment): The enviornment to be simulated.
        policy (PolicyBase): The policy used to generate actions.
        max_episode_steps (int): The maximum number of episodes.
        n_episodes (int): [description]
        to_visualize (bool, optional): Whether to visualize the enviorinment. Defaults to True.

    Returns:
        metrics (Sequence[Metrics]): The metrics of the all the episodes.
    """
    # TODO parallelize this piece of code.
    metrics: Sequence[Metrics] = list()
    for curr_eps in range(n_episodes):
        curr_metrics = simulate(env=env,
                                policy=policy,
                                max_episode_steps=max_episode_steps,
                                to_visualize=to_visualize)
        metrics.append(curr_metrics)
    return metrics
