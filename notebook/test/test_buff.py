from int_mpc.mdps.change_lane import ChangeLaneEnv
from drl_utils import buff_utils

if __name__ == "__main__":
    env = ChangeLaneEnv()
    buff = buff_utils.ReplayBuffer(1000)
    buff_utils.populate_replay_buffer(buff, env, env.get_random_policy(), 30,
                                      90)
    print(len(buff))
