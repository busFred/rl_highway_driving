#%%
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Sequence

from int_mpc.mdps.highway.change_lane import ChangeLaneEnv
from int_mpc.nets.train import dqn as dqn_train
from torch import nn

#%%
env = ChangeLaneEnv()

#%%
for _ in range(10):
    action, state, reward, is_terminal = env.step_random(to_visualize=True)
    if is_terminal:
        env.reset()
print("done")
# %%
