#%%
from typing import Dict
import gym
from highway_env.envs.highway_env import HighwayEnv

#%%
CONFIG: Dict = {
    "observation": {
        "type": "Kinematics",
        "features": ['presence', 'x', 'y', 'vx', 'vy'],
        "normalize": False,
        "observe_intentions": False
    }
}
env: HighwayEnv = gym.make("highway-v0")
env.configure(CONFIG)
env.reset()

#%%
for _ in range(100):
    env.render()
    action = env.action_space.sample()
    state, reward, is_terminal, info = env.step(action)
    if is_terminal is True:
        print("terminated")
        env.reset()
print("simulation end")

#%%
env.close()