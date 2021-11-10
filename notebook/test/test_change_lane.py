#%%
from int_mpc.mdps.change_lane import ChangeLaneEnv

#%%
env = ChangeLaneEnv()

#%%
state = env.reset()
policy = env.get_random_policy()
for _ in range(10):
    action = policy.sample_action(state)
    state, reward, is_terminal = env.step(action=action, to_visualize=True)
    if is_terminal:
        state = env.reset()
print("done")
# %%
