import csv
import multiprocessing as mp
import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import torch
from drl_algs import dqn as alg_dqn
from int_mpc.mdps.change_lane import (ChangeLaneConfig, ChangeLaneEnv,
                                      ChangeLaneMetrics)
from mdps import mdp_utils
from mdps.mdp_abc import Metrics

matplotlib.use("agg")


def create_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--env_config_path", type=str, required=True)
    parser.add_argument("--dqn_path",
                        type=str,
                        required=True,
                        help="path to the serialized dqn")
    parser.add_argument(
        "--export_metrics_dir",
        type=str,
        default=None,
        help="path to the directory containing exported metric")
    parser.add_argument("--n_test_episodes", type=int, default=100)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--to_vis", action="store_true")
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


def print_summary(summary: Dict[str, float]):
    for k in summary.keys():
        print(str.format("{}: ", k, summary[k]))


def main(args: Sequence[str]):
    argv: Namespace = parse_args(args)
    # create export path
    screenshot_dir_path: Optional[str] = None
    if argv.export_metrics_dir is not None:
        os.makedirs(argv.export_metrics_dir, exist_ok=True)
        screenshot_dir_path = os.path.join(argv.export_metrics_dir,
                                           "crash_screenshot")
        os.makedirs(screenshot_dir_path, exist_ok=True)
    # configure environment
    env_config = get_env_config(argv.env_config_path)
    env = ChangeLaneEnv(env_config)
    # create dqn
    device = torch.device("cuda") if argv.use_cuda else torch.device("cpu")
    net = torch.load(argv.model_path)
    policy = alg_dqn.DQN(env=env, dqn_net=net, device=device)
    # generate test metrics.
    metrics_l: Sequence[Metrics] = mdp_utils.simulate_episodes(
        env=env,
        policy=policy,
        max_episode_steps=env_config.max_episode_steps,
        n_episodes=argv.n_test_episodes,
        to_visualize=argv.to_vis,
        max_workers=argv.max_workers)
    for curr_sim_eps, metrics in enumerate(metrics_l):
        if isinstance(metrics, ChangeLaneMetrics)\
            and metrics.screenshot is not None\
            and argv.export_metrics_dir is not None\
            and screenshot_dir_path is not None:
            screenshot_path: str = os.path.join(
                screenshot_dir_path, str.format("eps_{}.png", curr_sim_eps))
            plt.imshow(metrics.screenshot)
            plt.savefig(screenshot_path)
            plt.close()
    # summarize metrics
    summary: Dict[str, float] = env.summarize_metrics_seq(metrics_l)
    print_summary(summary)
    # serialize metrics upon request
    if argv.export_metrics_dir is not None:
        metric_path: str = os.path.join(argv.export_metrics_dir, "metrics.pkl")
        pickle.dump(metrics_l, open(metric_path, "wb"))
        summary_path: str = os.path.join(argv.export_metrics_dir,
                                         "summary.csv")
        with open(summary_path, "w") as summary_file:
            writer = csv.DictWriter(summary_file, summary.keys())
            writer.writerow(summary)


if __name__ == "__main__":
    args: Sequence[str] = sys.argv[1:]
    mp.set_start_method("spawn")
    main(args=args)
