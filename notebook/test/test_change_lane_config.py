#%%
from int_mpc.mdps.change_lane import ChangeLaneConfig

#%%
CONFIG_STR: str = """
{
    "lane_counts": 4,
    "vehicles_count": 50,
    "initial_spacing": 1.0,
    "max_episode_steps": 30,
    "alpha": 0.4,
    "beta": -1.0,
    "reward_speed_range": [
        20.0,
        30.0
    ],
    "default_action": "faster"
}
"""

#%%
env_config: ChangeLaneConfig = ChangeLaneConfig.from_json(CONFIG_STR)
