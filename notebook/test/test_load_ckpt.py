#%%
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Sequence, Tuple, Union

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from attr import field
from drl_algs import dqn as alg_dqn
from int_mpc.mdps.change_lane import ChangeLaneConfig, ChangeLaneEnv
from int_mpc.nnet.change_lane.dqn import LinearDQN, LinearDQNConfig
from torch import multiprocessing as mp


def get_env_config(env_config_path: str):
    env_config: ChangeLaneConfig
    with open(env_config_path, "r") as config_file:
        env_config = ChangeLaneConfig.from_json(config_file.read())
    return env_config


#%%
class TrainDQNConfigs:
    # configs from json files
    env_config: ChangeLaneConfig
    dqn_config: alg_dqn.DQNConfig
    dqn_net_config: LinearDQNConfig
    # training configs
    n_val_episodes: int = field(default=20)
    lr: float = field(default=1e-3)
    # experiment name, version, and export path
    experiment_name: str
    experiments_root_path: str
    experiment_path: str  # experiments_root_path/experiment_name/version_{version}
    checkpoint_path: str  # experiment_path/checkpoints
    model_export_path_fn: str  # experiment_path/{experiment_name}.pt
    version: int
    # accelerate computing
    max_workers: Union[int, None]
    use_gpu: bool
    # loggers
    loggers: Tuple[pl_loggers.CSVLogger, pl_loggers.TensorBoardLogger]

    # experiments_root_path/experiment_name/version_{:02d}

    def __init__(self, args: Sequence[str]) -> None:
        argv = self._parse_args(args)
        self._load_configs(argv)
        self.n_val_episodes = argv.n_val_episodes
        self.lr = argv.lr
        self.experiment_name = argv.experiment_name
        self.experiments_root_path = argv.experiments_root_path
        self._configure_loggers()
        self._configure_export_paths()
        self.max_workers = argv.max_workers
        self.use_gpu = argv.use_gpu

    def _parse_args(self, args: Sequence[str]):
        parser = ArgumentParser()
        parser.add_argument("--env_config_path", type=str, required=True)
        parser.add_argument("--dqn_config_path", type=str, required=True)
        parser.add_argument("--dqn_net_config_path", type=str, required=True)
        parser.add_argument("--n_val_episodes", type=int, default=20)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--experiment_name", type=str, default="untitiled")
        parser.add_argument("--experiments_root_path",
                            type=str,
                            default="experiments")
        parser.add_argument("--max_workers", type=int, default=None)
        parser.add_argument("--use_gpu", action="store_true")
        argv = parser.parse_args(args)
        return argv

    def _load_configs(self, argv: Namespace):
        self.env_config = get_env_config(argv.env_config_path)
        self.dqn_config = self._get_dqn_config(argv.dqn_config_path)
        self.dqn_net_config = self._get_dqn_net_config(
            argv.dqn_net_config_path)

    def _get_dqn_config(self, dqn_config_path: str):
        dqn_config: alg_dqn.DQNConfig
        with open(dqn_config_path, "r") as config_file:
            dqn_config = alg_dqn.DQNConfig.from_json(config_file.read())
        return dqn_config

    def _get_dqn_net_config(self, dqn_net_config_path: str):
        net_config: LinearDQNConfig
        with open(dqn_net_config_path, "r") as config_file:
            net_config = LinearDQNConfig.from_json(config_file.read())
        return net_config

    def _configure_loggers(self):
        tfb_logger = pl_loggers.TensorBoardLogger(
            save_dir=self.experiments_root_path,
            name=self.experiment_name,
            sub_dir="tfb",
            default_hp_metric=False)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.experiments_root_path,
                                          name=self.experiment_name,
                                          version=tfb_logger.version)
        self.loggers = (csv_logger, tfb_logger)
        self.version = tfb_logger.version

    def _configure_export_paths(self):
        self.experiment_path = self.loggers[0].log_dir
        self.checkpoint_path = os.path.join(self.experiment_path,
                                            "checkpoints")
        self.model_export_path_fn = os.path.join(self.experiment_path,
                                                 self.experiment_name + ".pt")
        os.makedirs(self.experiment_path, exist_ok=True)


#%%
args = [
    "--env_config_path", "../../config/change_lane_d/env/01.json",
    "--dqn_config_path", "../../config/rl_alg/dqn/04.json",
    "--dqn_net_config_path", "../../config/change_lane_d/linear_dqn/00.json",
    "--n_val_episodes", "100", "--lr", "1e-5", "--experiment_name",
    "env_01_dqn_04_lnet_00_lr_1e5", "--experiments_root_path",
    "../../experiments"
]
# load configurations from command line arguments
configs = TrainDQNConfigs(args)
# create environment
env = ChangeLaneEnv(configs.env_config)
# create dqn_net
dqn_net = LinearDQN(configs.dqn_net_config)
dqn_net.init(env)
#  configure optimizer
optim = torch.optim.Adam(dqn_net.parameters(), lr=configs.lr)
# create dqn algorithm with dqn_net
ckpt_path_fn = "../../experiments/env_01_dqn_04_lnet_00_lr_1e5/version_0/checkpoints/env_01_dqn_04_lnet_00_lr_1e5_env_01_dqn_04_lnet_00_lr_1e5/0_0/checkpoints/epoch=2509-step=75299.ckpt"
dqn = alg_dqn.DQNTrain.load_from_checkpoint(
    ckpt_path_fn,
    env=env,
    dqn_net=dqn_net,
    dqn_config=configs.dqn_config,
    max_episode_steps=configs.env_config.max_episode_steps,
    n_val_episodes=configs.n_val_episodes,
    optimizer=optim,
    max_workers=configs.max_workers)

# %%

