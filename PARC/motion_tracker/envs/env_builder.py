import yaml

import parc.util.path_loader as path_loader
from parc.util.logger import Logger


def build_env(env_file, num_envs, device, visualize):
    env_config = path_loader.load_config(path_loader.resolve_path(env_file))

    env_name = env_config["env_name"]
    Logger.print("Building {} env".format(env_name))

    if env_name == "ig_parkour":
        import parc.motion_tracker.envs.ig_parkour.ig_parkour_env as ig_parkour_env
        env = ig_parkour_env.IGParkourEnv(config=env_config,
                                          num_envs=num_envs,
                                          device=device,
                                          visualize=visualize)
    elif env_name == "mj_parkour":
        import parc.motion_tracker.envs.mj_parkour_env as mj_parkour_env
        env = mj_parkour_env.MjParkourEnv(config=env_config,
                                           num_envs=num_envs,
                                           device=device,
                                           visualize=visualize)
    else:
        assert(False), "Unsupported env: {}".format(env_name)

    return env
