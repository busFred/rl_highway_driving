from typing import Sequence, TypeVar

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


StateTypeVar = TypeVar("StateTypeVar", bound=State)
ActionTypeVar = TypeVar("ActionTypeVar", bound=Action)
MetricsTypeVar = TypeVar("MetricsTypeVar", bound=Metrics)


def simulate(env: Environment[StateTypeVar, ActionTypeVar, MetricsTypeVar],
             policy: PolicyBase[StateTypeVar, ActionTypeVar],
             max_episode_steps: int,
             to_visualize: bool = True) -> MetricsTypeVar:
    """Simulate the MDP with the given policy.

    Args:
        env (Environment): The enviornment to be simulated.
        policy (PolicyBase): The policy used to generate actions.
        max_episode_steps (int): The maximum number of episodes.
        to_visualize (bool, optional): Whether to visualize the enviorinment. Defaults to True.

    Returns:
        metrics (Metrics): The metrics of the current episode.
    """
    state: StateTypeVar = env.reset()
    # step until timeout occurs
    for curr_step in range(max_episode_steps):
        action: ActionTypeVar = policy.sample_action(state)
        next_state, _, is_terminal = env.step(action=action,
                                              to_visualize=to_visualize)
        state = next_state
        if is_terminal:
            break
    metrics = env.calculate_metrics()
    return metrics


StateTypeVar = TypeVar("StateTypeVar", bound=State)
ActionTypeVar = TypeVar("ActionTypeVar", bound=Action)
MetricsTypeVar = TypeVar("MetricsTypeVar", bound=Metrics)


def simulate_episodes(env: Environment[StateTypeVar, ActionTypeVar, MetricsTypeVar],
                      policy: PolicyBase[StateTypeVar, ActionTypeVar],
                      max_episode_steps: int,
                      n_episodes: int,
                      to_visualize: bool = True) -> Sequence[MetricsTypeVar]:
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
    metrics: Sequence[MetricsTypeVar] = list()
    for curr_eps in range(n_episodes):
        curr_metrics = simulate(env=env,
                                policy=policy,
                                max_episode_steps=max_episode_steps,
                                to_visualize=to_visualize)
        metrics.append(curr_metrics)
    return metrics
