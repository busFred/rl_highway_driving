#%%
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from attr import field
from drl_algs import dqn as alg_dqn
from int_mpc.mdps.change_lane import ChangeLaneConfig, ChangeLaneEnv
from int_mpc.nnet.change_lane.dqn import LinearDQN, LinearDQNConfig
from torch import multiprocessing as mp

#%%
ckpt_path_fn = "../../experiments/"
dqn = alg_dqn.DQNTrain.load_from_checkpoint(ckpt_path_fn)