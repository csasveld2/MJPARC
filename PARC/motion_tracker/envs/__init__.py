"""Environment builders and definitions for the motion tracker.

This module provides environment implementations for motion tracking:

Isaac Gym Backend (original):
    - ig_env.py: Base Isaac Gym environment
    - ig_char_env.py: Character environment with PD control
    - ig_parkour/: Parkour and DeepMimic environments

mjlab Backend (new):
    - mj_char_env.py: Character environment using mjlab/MuJoCo Warp
    - mj_env_config.py: Configuration classes for mjlab environments

Usage (mjlab):
    from parc.motion_tracker.envs.mj_char_env import MJCharEnv
    from parc.motion_tracker.envs.mj_env_config import MJCharEnvCfg

    cfg = MJCharEnvCfg(
        char_file="path/to/humanoid.xml",
        motion_file="path/to/motions.yaml",
    )
    env = MJCharEnv(cfg=cfg)
"""

# mjlab exports (MuJoCo Warp backend)
try:
    from parc.motion_tracker.envs.mj_char_env import (
        MJCharEnv,
        PDExpActuator,
        PD1DActuator,
    )
    from parc.motion_tracker.envs.mj_env_config import (
        MJCharEnvCfg,
        MJParkourEnvCfg,
        ControlMode,
        CameraMode,
    )
    MJLAB_AVAILABLE = True
except ImportError:
    MJLAB_AVAILABLE = False

# Isaac Gym exports (original backend) - these may fail if Isaac Gym not installed
try:
    from parc.motion_tracker.envs.ig_env import IGEnv
    from parc.motion_tracker.envs.ig_char_env import IGCharEnv
    from parc.motion_tracker.envs.env_builder import build_env
    ISAACGYM_AVAILABLE = True
except ImportError:
    ISAACGYM_AVAILABLE = False