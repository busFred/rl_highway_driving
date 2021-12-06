#%%
from int_mpc.mdps.change_lane import ChangeLaneEnv, ChangeLaneConfig

#%%
CONFIG_PATH: str = "../../config/change_lane/01.json"
DQN_PATH: str = "../../model/env_01_dqn_03.pt"

#%%
env_config: ChangeLaneConfig
with open(CONFIG_PATH, "rb") as config_file:
    env_config = ChangeLaneConfig.from_json(config_file.read())
env = ChangeLaneEnv(env_config)

#%%
state = env.reset()
policy = env.get_random_policy()
for _ in range(10):
    action = policy.sample_action(state)
    action, state, reward, is_terminal = env.step(action=action,
                                                  to_visualize=True)
    if is_terminal:
        state = env.reset()
print("done")
# %%
