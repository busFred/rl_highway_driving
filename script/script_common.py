from int_mpc.mdps.change_lane import ChangeLaneConfig


def get_env_config(env_config_path: str):
    env_config: ChangeLaneConfig
    with open(env_config_path, "r") as config_file:
        env_config = ChangeLaneConfig.from_json(config_file.read())
    return env_config
