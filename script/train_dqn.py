import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Iterable, Sequence, Union

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from drl_algs import dqn as alg_dqn
from int_mpc.mdps.change_lane import ChangeLaneConfig, ChangeLaneEnv
from int_mpc.nnet.change_lane.dqn import LinearDQN
from pytorch_lightning.loggers.base import LightningLoggerBase


def create_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--env_config_path", type=str, required=True)
    parser.add_argument("--dqn_config_path", type=str, required=True)
    parser.add_argument("--n_val_episodes", type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--export_path", type=str, default=None)
    parser.add_argument("--metrics_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    return parser


def parse_args(args: Sequence[str]):
    parser = create_argparse()
    argv = parser.parse_args(args)
    return argv


def get_env_config(env_config_path: str):
    env_config: ChangeLaneConfig
    with open(env_config_path, "r") as config_file:
        env_config = ChangeLaneConfig.from_json(config_file.read())
    return env_config


def get_dqn_config(dqn_config_path: str):
    dqn_config: alg_dqn.DQNConfig
    with open(dqn_config_path, "r") as config_file:
        dqn_config = alg_dqn.DQNConfig.from_json(config_file.read())
    return dqn_config


def get_experiment_name(env_config_path: str, dqn_config_path: str) -> str:
    env_conf_filename: str = os.path.split(env_config_path)[-1]
    env_conf_filename = os.path.splitext(env_conf_filename)[0]
    dqn_conf_filename: str = os.path.split(dqn_config_path)[-1]
    dqn_conf_filename = os.path.splitext(dqn_conf_filename)[0]
    exp_name: str = str.format("env_{}_dqn_{}", env_conf_filename,
                               dqn_conf_filename)
    return exp_name


def main(args: Sequence[str]):
    argv: Namespace = parse_args(args)
    # create export path
    if argv.export_path is not None:
        os.makedirs(argv.export_path, exist_ok=True)
    if argv.metrics_path is not None:
        os.makedirs(argv.metrics_path, exist_ok=True)
    if argv.checkpoint_path is not None:
        os.makedirs(argv.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(argv.metrics_path, "csv"), exist_ok=True)
        os.makedirs(os.path.join(argv.metrics_path, "tfb"), exist_ok=True)
    # configure environment
    env_config: ChangeLaneConfig = get_env_config(argv.env_config_path)
    env = ChangeLaneEnv(env_config)
    # get dqn
    dqn_config: alg_dqn.DQNConfig = get_dqn_config(argv.dqn_config_path)
    net = LinearDQN()
    dqn = alg_dqn.DQNTrain(env=env,
                           dqn_net=net,
                           dqn_config=dqn_config,
                           max_episode_steps=env_config.max_episode_steps,
                           n_val_episodes=argv.n_val_episodes,
                           optimizer=torch.optim.Adam(net.parameters()),
                           max_workers=argv.max_workers)
    # configure experiment name
    exp_name: str = get_experiment_name(argv.env_config_path,
                                        argv.dqn_config_path)
    loggers: Union[LightningLoggerBase, Iterable[LightningLoggerBase],
                   bool] = False
    if argv.metrics_path is not None:
        loggers = [
            pl_loggers.CSVLogger(save_dir=os.path.join(argv.metrics_path,
                                                       "csv"),
                                 name=exp_name),
            pl_loggers.TensorBoardLogger(save_dir=os.path.join(
                argv.metrics_path, "tfb"),
                                         name=exp_name,
                                         default_hp_metric=False)
        ]
    # train the dqn
    trainer = pl.Trainer(max_epochs=dqn_config.n_episodes,
                         logger=loggers,
                         gpus=-1,
                         auto_select_gpus=True,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=5,
                         default_root_dir=argv.checkpoint_path)
    trainer.fit(dqn)
    # export model
    if argv.export_path is not None:
        model_path: str = os.path.join(argv.export_path, exp_name + ".pt")
        torch.save(dqn.dqn, model_path)


if __name__ == "__main__":
    args: Sequence[str] = sys.argv[1:]
    mp.set_start_method("spawn")
    main(args=args)
