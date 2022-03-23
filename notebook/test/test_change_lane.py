#%%
import os

import torch
from drl_algs.dqn import DQN
from int_mpc.mdps.change_lane import ChangeLaneConfig, ChangeLaneEnv
from int_mpc.nnet.change_lane.dqn import LinearDQN

#%%
CONFIG_PATH: str = "../../config/change_lane_d/env/02.json"
DQN_PATH: str = "../../model/env_02_dqn_04_ld_00.pt"
# GIF_EXPORT_PATH: str = "../../animation/"

#%%
# os.makedirs(GIF_EXPORT_PATH, exist_ok=True)
n_iters: int = 1

#%%
env_config: ChangeLaneConfig
with open(CONFIG_PATH, "rb") as config_file:
    env_config = ChangeLaneConfig.from_json(config_file.read())
env_config.max_episode_steps = 120
env = ChangeLaneEnv(env_config)

#%%
state = env.reset()
# policy = env.get_random_policy()
dqn_net: LinearDQN = torch.load(DQN_PATH)
policy = DQN(env, dqn_net)
for i in range(n_iters):
    for _ in range(env_config.max_episode_steps):
        action = policy.sample_action(state)
        action, state, reward, is_terminal = env.step(action=action,
                                                        to_visualize=True)
        if is_terminal:
            state = env.reset()
            break
env.reset()

# %%
