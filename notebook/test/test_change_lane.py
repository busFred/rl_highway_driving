#%%
from int_mpc.mdps.highway.change_lane import ChangeLaneEnv

#%%
env = ChangeLaneEnv()

#%%
for _ in range(10):
    action, state, reward, is_terminal = env.step_random(to_visualize=True)
    if is_terminal:
        env.reset()
print("done")
# %%
