import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Sequence

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from drl_algs import dqn as alg_dqn
from int_mpc.mdps.change_lane import ChangeLaneEnv
from int_mpc.nnet.change_lane.dqn import LinearDQN


def create_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dqn_config_path", type=str, required=True)
    parser.add_argument("--export_path", type=str, required=True)
    parser.add_argument("--vehicles_count", type=int, default=200)
    parser.add_argument("--n_test_episodes", type=int, default=1000)
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
    # get configuration
    dqn_config: alg_dqn.DQNConfig = get_config(argv.dqn_config_path)
    dqn = alg_dqn.DQNTrain(env=env,
                           dqn_net=net,
                           dqn_config=dqn_config,
                           optimizer=torch.optim.Adam(net.parameters()))
    config_filename: str = os.path.split(argv.dqn_config_path)[-1]
    config_filename = os.path.splitext(config_filename)[0]
    loggers = [
        pl_loggers.TensorBoardLogger(save_dir=argv.export_path,
                                     name=config_filename + "_tfb",
                                     default_hp_metric=False),
        pl_loggers.CSVLogger(save_dir=argv.export_path,
                             name=config_filename + "_csv")
    ]
    trainer = pl.Trainer(logger=loggers,
                         gpus=-1,
                         auto_select_gpus=True,
                         check_val_every_n_epoch=5)

    trainer.fit(dqn)
    # export model
    model_path: str = os.path.join(argv.export_path, "model.pt")
    torch.save(dqn.dqn, model_path)


if __name__ == "__main__":
    args: Sequence[str] = sys.argv[1:]
    main(args=args)
