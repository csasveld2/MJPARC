"""Environment builders and definitions for the motion tracker.

This module provides environment implementations for motion tracking:

mjlab Backend (new, recommended):
    - mj_env_config.py: Environment configuration factory
    - mj_mdp.py: Custom MDP components (placeholder for motion tracking)
    - mj_launch_example.py: Example launch script

Isaac Gym Backend (original, legacy):
    - ig_env.py: Base Isaac Gym environment
    - ig_char_env.py: Character environment with PD control
    - ig_parkour/: Parkour and DeepMimic environments

Usage (mjlab):
    from parc.motion_tracker.envs.mj_env_config import make_parc_tracking_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_parc_tracking_env_cfg(
        char_file="data/assets/humanoid.xml",
        num_envs=4096,
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")

Command line:
    uv run PARC/motion_tracker/envs/mj_launch_example.py \\
        --char-file data/assets/humanoid.xml \\
        --num-envs 4096 \\
        --device cuda:0
"""

# mjlab exports
MJLAB_AVAILABLE = False
try:
    from parc.motion_tracker.envs.mj_env_config import make_parc_tracking_env_cfg
    from parc.motion_tracker.envs import mj_mdp
    MJLAB_AVAILABLE = True
except ImportError:
    pass

# Isaac Gym exports (original backend)
ISAACGYM_AVAILABLE = False
try:
    from parc.motion_tracker.envs.ig_env import IGEnv
    from parc.motion_tracker.envs.ig_char_env import IGCharEnv
    from parc.motion_tracker.envs.env_builder import build_env
    ISAACGYM_AVAILABLE = True
except ImportError:
    pass
