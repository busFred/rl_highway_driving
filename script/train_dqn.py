import math
import pickle
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import List, Sequence, Union

import torch
from drl_algs import dqn as dqn_train
from drl_utils.buff_utils import ReplayBuffer
from int_mpc.mdps.highway.change_lane import ChangeLaneEnv
from int_mpc.mdps.highway.highway_mdp import (HighwayEnvDiscreteAction,
                                              HighwayEnvState)
from torch import nn


@dataclass
class ChangeLaneMetric:
    distance_travel: float = field()
    terminated_crash: bool = field()
    n_episodes_to_crash: float = field()


def create_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dqn_config_path", type=str, required=True)
    return parser


def parse_args(args: Sequence[str]):
    parser = create_argparse()
    argv = parser.parse_args(args)
    return argv


def get_config(argv: Namespace):
    dqn_config_path: str = argv.dqn_config_path
    dqn_config: dqn_train.DQNConfig
    with open(dqn_config_path, "r") as config_file:
        dqn_config = dqn_train.DQNConfig.from_json(config_file.read())
    return dqn_config


def get_value_net() -> nn.Module:
    model = nn.Sequential(nn.Flatten(1, -1), nn.Linear(28, 100), nn.ReLU(),
                          nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Linear(100, len(HighwayEnvDiscreteAction)))
    return model


def simulate(env: ChangeLaneEnv,
             dqn: dqn_train.DQN,
             dqn_config: dqn_train.DQNConfig,
             to_vis: bool = True) -> ChangeLaneMetric:
    dqn.eval()
    state: HighwayEnvState = env.reset()
    start_loc: float = state.observation[0, 0]
    curr_eps: int = 0
    # step until timeout occurs
    while curr_eps < dqn_config.max_episode_steps:
        _, next_state, _, is_terminal = dqn_train.eps_greedy_step(
            env=env,
            state=state,
            dqn=dqn,
            dqn_config=dqn_config,
            to_visualize=to_vis)
        state = next_state
        curr_eps = curr_eps + 1
        if is_terminal:
            break
    end_loc: float = state.observation[0, 0]
    distance_travel: float = end_loc - start_loc
    terminated_crash: bool = state.is_crashed
    n_episodes_to_crash: int = curr_eps if terminated_crash else -1
    metric = ChangeLaneMetric(distance_travel=distance_travel,
                              terminated_crash=terminated_crash,
                              n_episodes_to_crash=n_episodes_to_crash)
    return metric


def main(args: Sequence[str]):
    # argv: Namespace = parse_args(args)
    # dqn_config: dqn_train.DQNConfig = get_config(argv)
    env = ChangeLaneEnv()
    net = get_value_net()
    dqn = dqn_train.DQN(dqn=net, optimizer=torch.optim.Adam(net.parameters()))
    # configure n_episode
    dqn_config = dqn_train.DQNConfig(max_buff_size=200)
    n_eps: int = 10
    n_iters: int = math.ceil(dqn_config.n_episodes / 10)
    dqn_config.n_episodes = n_eps
    replay_buffer: Union[ReplayBuffer, None] = None
    metrics: List[List[ChangeLaneMetric]] = list()
    for curr_eps in range(n_iters):
        replay_buffer = dqn_train.train_dqn(env=env,
                                            dqn=dqn,
                                            dqn_config=dqn_config,
                                            replay_buffer=replay_buffer)
        curr_metrics: List[ChangeLaneMetric] = list()
        for curr_sim_eps in range(n_eps):
            metric = simulate(env=env,
                              dqn=dqn,
                              dqn_config=dqn_config,
                              to_vis=True)
            curr_metrics.append(metric)
        metrics.append(curr_metrics)
    pickle.dump(metrics, open("metrics.pkl", "wb"))
    torch.save(dqn.dqn, "model.pt")


if __name__ == "__main__":
    args: Sequence[str] = sys.argv[1:]
    main(args=args)
