#%%
import torch
from drl_algs.dqn import DQN
from int_mpc.mdps.change_lane import ChangeLaneConfig, ChangeLaneEnv
from mdps import mdp_utils

#%%
CONFIG_PATH: str = "../config/change_lane/01.json"
DQN_PATH: str = "../model/env_01_dqn_02.pt"

#%%
env_config: ChangeLaneConfig
with open(CONFIG_PATH, "rb") as config_file:
    env_config = ChangeLaneConfig.from_json(config_file.read())
env = ChangeLaneEnv(env_config)

#%%
dqn_net = torch.load(DQN_PATH)
dqn: DQN = DQN(env, dqn_net)

#%%
metric = mdp_utils.simulate(env, dqn, max_episode_steps=30, to_visualize=True)
