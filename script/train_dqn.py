import sys
from argparse import ArgumentParser, Namespace
from typing import Sequence

import torch
from drl_algs import dqn as dqn_train
from int_mpc.mdps.highway.change_lane import ChangeLaneEnv
from int_mpc.mdps.highway.highway_mdp import HighwayEnvDiscreteAction
from torch import nn


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
    model = nn.Sequential(nn.Flatten(1, -1), nn.Linear(20, 100), nn.ReLU(),
                          nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Linear(100, len(HighwayEnvDiscreteAction)))
    return model


def main(args: Sequence[str]):
    # argv: Namespace = parse_args(args)
    # dqn_config: dqn_train.DQNConfig = get_config(argv)
    dqn_config = dqn_train.DQNConfig(max_buff_size=200)
    env = ChangeLaneEnv()
    net = get_value_net()
    dqn = dqn_train.DQN(dqn=net, optimizer=torch.optim.Adam(net.parameters()))
    dqn_train.train_dqn(env=env,
                        dqn=dqn,
                        dqn_config=dqn_config,
                        to_visualize=True)


if __name__ == "__main__":
    args: Sequence[str] = sys.argv[1:]
    main(args=args)
