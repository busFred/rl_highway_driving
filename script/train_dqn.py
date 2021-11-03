import math
import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import List, Sequence

import torch
from drl_algs import dqn as dqn_train
from int_mpc.mdps.highway.change_lane import ChangeLaneEnv
from int_mpc.mdps.highway.highway_mdp import (HighwayEnvDiscreteAction,
                                              HighwayEnvState)
from int_mpc.nnet.dqn import LinearDQN
from torch import nn


@dataclass
class ChangeLaneMetric:
    distance_travel: float = field()
    terminated_crash: bool = field()
    n_steps_to_crash: float = field()


def create_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dqn_config_path", type=str, required=True)
    parser.add_argument("--export_path", type=str, required=True)
    parser.add_argument("--vehicles_count", type=int, default=200)
    parser.add_argument("--n_test_episodes", type=int, default=1000)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--to_vis", action="store_true")
    return parser


def parse_args(args: Sequence[str]):
    parser = create_argparse()
    argv = parser.parse_args(args)
    return argv


def get_config(dqn_config_path: str):
    dqn_config: dqn_train.DQNConfig
    with open(dqn_config_path, "r") as config_file:
        dqn_config = dqn_train.DQNConfig.from_json(config_file.read())
    return dqn_config


# def get_value_net() -> nn.Module:
#     model = nn.Sequential(nn.Flatten(1, -1), nn.Linear(28, 100), nn.ReLU(),
#                           nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 100),
#                           nn.ReLU(),
#                           nn.Linear(100, len(HighwayEnvDiscreteAction)))
#     return model


def simulate(env: ChangeLaneEnv,
             dqn: dqn_train.DQN,
             dqn_config: dqn_train.DQNConfig,
             to_vis: bool = True) -> ChangeLaneMetric:
    dqn.eval()
    state: HighwayEnvState = env.reset()
    start_loc: float = state.observation[0, 0]
    total_step: int = 0
    # step until timeout occurs
    for curr_step in range(dqn_config.max_episode_steps):
        _, next_state, _, is_terminal = dqn_train.greedy_step(env=env,
                                                              state=state,
                                                              dqn=dqn,
                                                              to_vis=to_vis)
        state = next_state
        total_step = curr_step + 1
        if is_terminal:
            break
    end_loc: float = state.observation[0, 0]
    distance_travel: float = end_loc - start_loc
    terminated_crash: bool = state.is_crashed
    n_steps_to_crash: int = total_step if terminated_crash else -1
    metric = ChangeLaneMetric(distance_travel=distance_travel,
                              terminated_crash=terminated_crash,
                              n_steps_to_crash=n_steps_to_crash)
    return metric


def main(args: Sequence[str]):
    argv: Namespace = parse_args(args)
    # create export path
    os.makedirs(argv.export_path, exist_ok=True)
    # configure environment
    env = ChangeLaneEnv(vehicles_count=argv.vehicles_count)
    # create dqn
    net = LinearDQN()
    device = torch.device("cuda") if argv.use_cuda else torch.device("cpu")
    dqn = dqn_train.DQN(dqn=net,
                        optimizer=torch.optim.Adam(net.parameters()),
                        device=device)
    # get configuration
    dqn_config: dqn_train.DQNConfig = get_config(argv.dqn_config_path)
    # train agent
    dqn_train.train_dqn(env=env, dqn=dqn, dqn_config=dqn_config)
    # export model
    model_path: str = os.path.join(argv.export_path, "model.pt")
    torch.save(dqn.dqn, model_path)
    # generate test metrics.
    metrics: List[ChangeLaneMetric] = list()
    for curr_sim_eps in range(argv.n_test_episodes):
        print(str.format("val {}/{}", curr_sim_eps, argv.n_test_episodes))
        metric = simulate(env=env,
                          dqn=dqn,
                          dqn_config=dqn_config,
                          to_vis=argv.to_vis)
        metrics.append(metric)
    # serialize metrics
    metric_path: str = os.path.join(argv.export_path, "metrics.pkl")
    pickle.dump(metrics, open(metric_path, "wb"))


if __name__ == "__main__":
    args: Sequence[str] = sys.argv[1:]
    main(args=args)
