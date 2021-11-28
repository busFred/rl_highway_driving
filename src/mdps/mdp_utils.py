import copy
import multiprocessing as mp
from typing import Optional, Sequence

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
        _, next_state, _, is_terminal = env.step(action=action,
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
                      to_visualize: bool = True,
                      max_workers: Optional[int] = None,
                      timeout: float = 15 * 60) -> Sequence[Metrics]:
    """Simulate the MDP with the given policy.

    Args:
        env (Environment): The enviornment to be simulated.
        policy (PolicyBase): The policy used to generate actions.
        max_episode_steps (int): The maximum number of episodes.
        n_episodes (int): Number of episodes.
        to_visualize (bool, optional): Whether to visualize the enviorinment. Defaults to True.
        max_workers (int): Maximum process to use.
        timeout (float): Amount of time to kill the child processes if it returns nothing.

    Returns:
        metrics (Sequence[Metrics]): The metrics of the all the episodes.
    """
    with mp.get_context("spawn").Pool(processes=max_workers) as pool:
        metrics: Sequence[Metrics] = list()
        result = pool.starmap_async(
            simulate, [(env.new_env_like(), copy.deepcopy(policy),
                        max_episode_steps, to_visualize)
                       for _ in range(n_episodes)])
        try:
            metrics = result.get(timeout=timeout)
        except TimeoutError as te:
            raise te
    return metrics
