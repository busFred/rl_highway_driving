import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Sequence

import torch
from drl_algs import dqn as alg_dqn
from int_mpc.mdps.highway.change_lane import ChangeLaneEnv
from int_mpc.nnet.change_lane.dqn import LinearDQN


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
    dqn_config: alg_dqn.DQNConfig
    with open(dqn_config_path, "r") as config_file:
        dqn_config = alg_dqn.DQNConfig.from_json(config_file.read())
    return dqn_config


def main(args: Sequence[str]):
    argv: Namespace = parse_args(args)
    # create export path
    os.makedirs(argv.export_path, exist_ok=True)
    # configure environment
    env = ChangeLaneEnv(vehicles_count=argv.vehicles_count)
    # create dqn
    net = LinearDQN()
    device = torch.device("cuda") if argv.use_cuda else torch.device("cpu")
    dqn = alg_dqn.DQN(dqn=net,
                      optimizer=torch.optim.Adam(net.parameters()),
                      device=device)
    # get configuration
    dqn_config: alg_dqn.DQNConfig = get_config(argv.dqn_config_path)
    # train agent
    alg_dqn.train_dqn(env=env, dqn=dqn, dqn_config=dqn_config)
    # export model
    model_path: str = os.path.join(argv.export_path, "model.pt")
    torch.save(dqn.dqn, model_path)


if __name__ == "__main__":
    args: Sequence[str] = sys.argv[1:]
    main(args=args)
