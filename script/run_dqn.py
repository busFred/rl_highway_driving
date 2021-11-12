import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from drl_algs import dqn as alg_dqn
from int_mpc.mdps.change_lane import ChangeLaneEnv, ChangeLaneMetrics
from mdps.mdp_utils import simulate

matplotlib.use("agg")


def create_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="path to the serialized dqn")
    parser.add_argument(
        "--export_metrics_dir",
        type=str,
        default=None,
        help="path to the directory containing exported metric")
    parser.add_argument("--vehicles_count", type=int, default=200)
    parser.add_argument("--max_episode_steps", type=int, default=50)
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
    screenshot_dir_path: Optional[str] = None
    if argv.export_metrics_dir is not None:
        os.makedirs(argv.export_metrics_dir, exist_ok=True)
        screenshot_dir_path = os.path.join(argv.export_metrics_dir,
                                           "crash_screenshot")
        os.makedirs(screenshot_dir_path, exist_ok=True)
    # configure environment
    env = ChangeLaneEnv(vehicles_count=argv.vehicles_count)
    # create dqn
    net = torch.load(argv.model_path)
    device = torch.device("cuda") if argv.use_cuda else torch.device("cpu")
    dqn = alg_dqn.DQN(dqn=net, device=device)
    policy = alg_dqn.GreedyDQNPolicy(env=env, dqn=dqn)
    # generate test metrics.
    metrics_l: List[ChangeLaneMetrics] = list()
    for curr_sim_eps in range(argv.n_test_episodes):
        print(str.format("val {}/{}", curr_sim_eps + 1, argv.n_test_episodes))
        metrics: ChangeLaneMetrics = simulate(
            env=env,
            policy=policy,
            max_episode_steps=argv.max_episode_steps,
            to_visualize=argv.to_vis)
        if metrics.screenshot is not None and argv.export_metrics_dir is not None and screenshot_dir_path is not None:
            screenshot_path: str = os.path.join(
                screenshot_dir_path, str.format("eps_{}.png", curr_sim_eps))
            plt.imshow(metrics.screenshot)
            plt.savefig(screenshot_path)
            plt.close()
        metrics_l.append(metrics)
    # serialize metrics
    if argv.export_metrics_dir is not None:
        metric_path: str = os.path.join(argv.export_metrics_dir, "metrics.pkl")
        pickle.dump(metrics_l, open(metric_path, "wb"))


if __name__ == "__main__":
    args: Sequence[str] = sys.argv[1:]
    main(args=args)
