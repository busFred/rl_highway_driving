#%%
import csv
import fnmatch
import itertools
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

#%%
metrics_path: str = "../../metrics/train/csv"
exp_paths: List[str] = os.listdir(metrics_path)
plot_root_path: str = "output"
os.makedirs(plot_root_path, exist_ok=True)


#%%
def process_experiment(exp_path: str):
    # (n_versions, )
    file_paths: List[str] = list()
    data: List[Dict[str, List]] = list()
    for ver_path in os.listdir(exp_path):
        ver_path = os.path.join(exp_path, ver_path)
        if not os.path.isdir(ver_path):
            continue
        # assume only one *.csv in an experiment/version
        csv_filename: str = fnmatch.filter(os.listdir(ver_path), "*.csv")[0]
        csv_path: str = os.path.join(ver_path, csv_filename)
        with open(csv_path, "r") as csv_file:
            reader = csv.DictReader(csv_file, restval=None)
            if reader.fieldnames is None:
                continue
            curr_data: Dict[str, List[Any]] = dict()
            for field in reader.fieldnames:
                curr_data[field] = list()
            for row in reader:
                for field in reader.fieldnames:
                    val = None
                    try:
                        val = float(row[field])
                    except ValueError:
                        val = None
                    curr_data[field].append(val)
            data.append(curr_data)
        file_paths.append(csv_path)
    return file_paths, data


#%%
def extract_field(exp_data: List[Dict[str, List]], file_paths: List[str],
                  field: str):
    exp_paths = list()
    values = list()
    for data, file_path in zip(exp_data, file_paths):
        if field not in data.keys():
            continue
        values.append(list(filter(lambda x: x is not None, data[field])))
        exp_paths.append(file_path)
    return exp_paths, values


#%%
file_paths: List[str] = list()
exp_data: List[Dict[str, List]] = list()
for exp_path in exp_paths:
    exp_path = os.path.join(metrics_path, exp_path)
    if not os.path.isdir(exp_path):
        continue
    ps, ds = process_experiment(exp_path)
    file_paths.extend(ps)
    exp_data.extend(ds)

#%%
# plot training loss
loss_paths, loss_vals = extract_field(exp_data, file_paths, "training_loss")
for curr_path, loss in zip(loss_paths, loss_vals):
    with plt.style.context("science", after_reset=True):
        plt.plot(list(range(0, len(loss))),
                 loss,
                 color='black',
                 linewidth=1,
                 markersize=3)
        plot_filename = str.format("{}_{}_{}.png",
                                   Path(curr_path).parts[-3],
                                   Path(curr_path).parts[-2], "loss")
        plot_path = os.path.join(plot_root_path, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.show()

# %%
reward_paths, reward_vals = extract_field(exp_data, file_paths,
                                          "avg_total_reward")
lrs = ["1e-3", "1e-5", "1e-5", "1e-3"]
with plt.style.context(["science", "ieee", "std-colors"], after_reset=True):
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for curr_path, rewards, color in zip(reward_paths, reward_vals,
                                         itertools.cycle(cycle)):
        plt.plot(list(range(1, 5 * len(rewards) + 1, 5)),
                 rewards,
                 color=color,
                 linewidth=1,
                 markersize=3,
                 alpha=0.3)
    for curr_path, rewards, lr, color in zip(reward_paths, reward_vals, lrs,
                                             itertools.cycle(cycle)):
        exp_conf = str.format("{}_lr_{}", Path(curr_path).parts[-3], lr)
        exp_conf = exp_conf.replace("_", "\\_")
        plt.plot(list(range(1, 5 * len(rewards) + 1, 5)),
                 uniform_filter1d(rewards, size=100),
                 label=exp_conf,
                 color=color,
                 linewidth=1,
                 markersize=3)
    plot_filename = "reward_overlap"
    plot_path = os.path.join(plot_root_path, plot_filename)
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
    cf = plt.gcf()
    cf.set_size_inches(cf.get_size_inches()[0] * 2, cf.get_size_inches()[1])
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

# %%
dist_paths, dist_vals = extract_field(exp_data, file_paths, "avg_distance")
lrs = ["1e-3", "1e-5", "1e-5", "1e-3"]
with plt.style.context(["science", "ieee", "std-colors"], after_reset=True):
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for curr_path, dists, color in zip(dist_paths, dist_vals,
                                       itertools.cycle(cycle)):
        plt.plot(list(range(1, 5 * len(dists) + 1, 5)),
                 dists,
                 color=color,
                 linewidth=1,
                 markersize=3,
                 alpha=0.3)
    for curr_path, dists, lr, color in zip(dist_paths, dist_vals, lrs,
                                           itertools.cycle(cycle)):
        exp_conf = str.format("{}_lr_{}", Path(curr_path).parts[-3], lr)
        exp_conf = exp_conf.replace("_", "\\_")
        plt.plot(list(range(1, 5 * len(dists) + 1, 5)),
                 uniform_filter1d(dists, size=100),
                 label=exp_conf,
                 color=color,
                 linewidth=1,
                 markersize=3)
    plt.axhline(y=308,
                xmin=0,
                color='grey',
                linestyle='--',
                linewidth=1,
                label="random")
    plot_filename = "distance_overlap"
    plot_path = os.path.join(plot_root_path, plot_filename)
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
    cf = plt.gcf()
    cf.set_size_inches(cf.get_size_inches()[0] * 2, cf.get_size_inches()[1])
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

# %%
col_paths, col_vals = extract_field(exp_data, file_paths, "n_crashes")
lrs = ["1e-3", "1e-5", "1e-5", "1e-3"]
with plt.style.context(["science", "ieee", "std-colors"], after_reset=True):
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for curr_path, cols, color in zip(col_paths, col_vals,
                                      itertools.cycle(cycle)):
        plt.plot(list(range(1, 5 * len(cols) + 1, 5)),
                 cols,
                 color=color,
                 linewidth=1,
                 markersize=3,
                 alpha=0.3)
    for curr_path, cols, lr, color in zip(col_paths, col_vals, lrs,
                                          itertools.cycle(cycle)):
        exp_conf = str.format("{}_lr_{}", Path(curr_path).parts[-3], lr)
        exp_conf = exp_conf.replace("_", "\\_")
        plt.plot(list(range(1, 5 * len(cols) + 1, 5)),
                 uniform_filter1d(cols, size=100),
                 label=exp_conf,
                 color=color,
                 linewidth=1,
                 markersize=3)
    plt.axhline(y=18,
                xmin=0,
                color='grey',
                linestyle='--',
                linewidth=1,
                label="random")
    plot_filename = "collision_overlap"
    plot_path = os.path.join(plot_root_path, plot_filename)
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
    cf = plt.gcf()
    cf.set_size_inches(cf.get_size_inches()[0] * 2, cf.get_size_inches()[1])
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

# %%
steps_paths, steps_vals = extract_field(exp_data, file_paths, "avg_steps_to_crash")
lrs = ["1e-3", "1e-5", "1e-5", "1e-3"]
with plt.style.context(["science", "ieee", "std-colors"], after_reset=True):
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for curr_path, steps, color in zip(steps_paths, steps_vals,
                                       itertools.cycle(cycle)):
        plt.plot(list(range(1, 5 * len(steps) + 1, 5)),
                 steps,
                 color=color,
                 linewidth=1,
                 markersize=3,
                 alpha=0.3)
    for curr_path, steps, lr, color in zip(steps_paths, steps_vals, lrs,
                                           itertools.cycle(cycle)):
        exp_conf = str.format("{}_lr_{}", Path(curr_path).parts[-3], lr)
        exp_conf = exp_conf.replace("_", "\\_")
        plt.plot(list(range(1, 5 * len(steps) + 1, 5)),
                 uniform_filter1d(steps, size=100),
                 label=exp_conf,
                 color=color,
                 linewidth=1,
                 markersize=3)
    plt.axhline(y=10,
                xmin=0,
                color='grey',
                linestyle='--',
                linewidth=1,
                label="random")
    plot_filename = "steps_to_crash_overlap"
    plot_path = os.path.join(plot_root_path, plot_filename)
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
    cf = plt.gcf()
    cf.set_size_inches(cf.get_size_inches()[0] * 2, cf.get_size_inches()[1])
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

# %%
