"""
Example script showing how to launch the PARC environment with mjlab.

This demonstrates the translation from Isaac Gym to mjlab:

Isaac Gym (original):
    from parc.motion_tracker.envs.env_builder import build_env
    env = build_env(config, num_envs=4096, device="cuda:0", visualize=True)

mjlab (new):
    from parc.motion_tracker.envs.mj_char_env import MJCharEnv
    from parc.motion_tracker.envs.mj_env_config import MJCharEnvCfg, MJParkourEnvCfg
    cfg = MJCharEnvCfg(...)
    env = MJCharEnv(cfg=cfg)

Key Differences in Tensor Access:

    +----------------------------+----------------------------------------+
    | Isaac Gym (gymtorch)       | mjlab (scene.data)                     |
    +----------------------------+----------------------------------------+
    | gym.acquire_actor_root_    | env.scene["robot"].data.root_pos_w     |
    | state_tensor()             |                                        |
    | + gymtorch.wrap_tensor()   |                                        |
    +----------------------------+----------------------------------------+
    | self._char_root_pos        | env.char_root_pos (property)           |
    | self._char_root_rot        | env.char_root_rot (property)           |
    | self._char_dof_pos         | env.char_dof_pos (property)            |
    | self._char_dof_vel         | env.char_dof_vel (property)            |
    | self._char_contact_forces  | env.char_contact_forces (property)     |
    +----------------------------+----------------------------------------+

    The key insight is that mjlab exposes MuJoCo Warp's internal buffers
    directly as PyTorch tensors through the scene.data interface.

Usage:
    # Basic usage
    python mj_launch_example.py --char-file path/to/humanoid.xml

    # With motion tracking
    python mj_launch_example.py --char-file humanoid.xml --motion-file motions.yaml

    # Full training
    python mj_launch_example.py --char-file humanoid.xml --motion-file motions.yaml --num-envs 4096
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch PARC motion tracking environment with mjlab"
    )

    # Basic settings
    parser.add_argument(
        "--char-file",
        type=str,
        required=True,
        help="Path to character MJCF XML file",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default="",
        help="Path to motion library file (YAML or pickle)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for simulation (cuda:N or cpu)",
    )

    # Environment settings
    parser.add_argument(
        "--episode-length",
        type=float,
        default=10.0,
        help="Episode length in seconds",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_exp",
        choices=["pd", "vel", "torque", "pd_exp", "pd_1d"],
        help="Control mode for the character",
    )

    # Visualization
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable visualization",
    )
    parser.add_argument(
        "--camera-mode",
        type=str,
        default="track",
        choices=["still", "track"],
        help="Camera tracking mode",
    )

    # Training settings
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of environment steps to run",
    )
    parser.add_argument(
        "--print-obs",
        action="store_true",
        help="Print observation shapes",
    )

    return parser.parse_args()


def build_env_config(args) -> "MJCharEnvCfg":
    """
    Build environment configuration from command line arguments.

    This shows how to configure the mjlab environment similar to how
    Isaac Gym environments were configured via YAML files.

    Returns:
        MJCharEnvCfg: Environment configuration
    """
    # Import configuration classes
    from parc.motion_tracker.envs.mj_env_config import (
        MJCharEnvCfg,
        MJParkourEnvCfg,
        ControlMode,
        CameraMode,
    )

    # Create base configuration
    if args.motion_file:
        # Use parkour config for motion tracking
        cfg = MJParkourEnvCfg()
    else:
        # Use basic character config
        cfg = MJCharEnvCfg()

    # Set character file
    cfg.char_file = os.path.abspath(args.char_file)

    # Set scene parameters
    cfg.scene.num_envs = args.num_envs

    # Set episode settings
    cfg.episode_length_s = args.episode_length

    # Set control mode
    cfg.control_mode = ControlMode[args.control_mode]

    # Set camera mode
    cfg.camera_mode = CameraMode[args.camera_mode]

    # Set motion file if provided
    if args.motion_file:
        cfg.motion_file = os.path.abspath(args.motion_file)

    # Configure key bodies (typical humanoid)
    cfg.key_bodies = [
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
    ]

    # Set reward weights (DeepMimic defaults)
    cfg.pose_w = 0.5
    cfg.vel_w = 0.1
    cfg.root_pos_w = 0.15
    cfg.root_vel_w = 0.1
    cfg.key_pos_w = 0.15

    return cfg


def run_environment(args):
    """
    Run the mjlab environment.

    This demonstrates the basic environment loop which is similar to
    Isaac Gym but uses mjlab's ManagerBasedRLEnv interface.
    """
    # Import environment
    from parc.motion_tracker.envs.mj_char_env import MJCharEnv

    # Build configuration
    cfg = build_env_config(args)

    print("=" * 60)
    print("PARC mjlab Environment")
    print("=" * 60)
    print(f"Character file: {cfg.char_file}")
    print(f"Motion file: {cfg.motion_file or 'None'}")
    print(f"Num envs: {cfg.scene.num_envs}")
    print(f"Control mode: {cfg.control_mode}")
    print(f"Episode length: {cfg.episode_length_s}s")
    print("=" * 60)

    # Create environment
    render_mode = "human" if args.render else None
    env = MJCharEnv(cfg=cfg, render_mode=render_mode)

    # Print observation space
    obs_space = env.observation_space
    action_space = env.action_space
    print(f"\nObservation space: {obs_space}")
    print(f"Action space: {action_space}")

    # Print tensor mapping information
    print("\n" + "=" * 60)
    print("Tensor Access Translation (Isaac Gym → mjlab)")
    print("=" * 60)
    print("""
    Isaac Gym (gymtorch):
        root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        self._char_root_pos = root_state[:, 0, 0:3]
        self._char_root_rot = root_state[:, 0, 3:7]
        self._char_dof_pos = dof_state[:, :num_dofs, 0]
        self._char_dof_vel = dof_state[:, :num_dofs, 1]

    mjlab (MuJoCo Warp):
        # Direct tensor access via scene.data or articulation.data
        root_pos = env.scene["robot"].data.root_pos_w      # [num_envs, 3]
        root_rot = env.scene["robot"].data.root_quat_w     # [num_envs, 4]
        dof_pos = env.scene["robot"].data.joint_pos        # [num_envs, num_dofs]
        dof_vel = env.scene["robot"].data.joint_vel        # [num_envs, num_dofs]
        contacts = env.scene["robot"].data.net_contact_forces_w  # [num_envs, num_bodies, 3]

        # Via environment properties (recommended):
        root_pos = env.char_root_pos
        root_rot = env.char_root_rot
        dof_pos = env.char_dof_pos
        dof_vel = env.char_dof_vel
        contacts = env.char_contact_forces
    """)

    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    if args.print_obs:
        print(f"\nObservation tensor:\n{obs}")

    # Run environment loop
    print(f"\nRunning {args.num_steps} steps...")
    total_reward = torch.zeros(cfg.scene.num_envs, device=env.device)

    for step in range(args.num_steps):
        # Sample random action (or use policy)
        action = torch.zeros(
            cfg.scene.num_envs,
            action_space.shape[0],
            device=env.device
        )
        # Add small random noise for testing
        action += 0.1 * torch.randn_like(action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print progress
        if (step + 1) % 100 == 0:
            mean_reward = total_reward.mean().item() / (step + 1)
            print(f"Step {step + 1}/{args.num_steps}, Mean reward: {mean_reward:.4f}")

        # Handle resets
        done = terminated | truncated
        if done.any():
            # Reset is handled automatically by ManagerBasedRLEnv
            pass

    # Print final statistics
    print("\n" + "=" * 60)
    print("Run Complete")
    print("=" * 60)
    print(f"Total steps: {args.num_steps}")
    print(f"Mean total reward: {total_reward.mean().item():.4f}")
    print(f"Max total reward: {total_reward.max().item():.4f}")
    print(f"Min total reward: {total_reward.min().item():.4f}")

    # Close environment
    env.close()


def main():
    """Main entry point."""
    args = parse_args()

    # Check if character file exists
    if not os.path.exists(args.char_file):
        print(f"Error: Character file not found: {args.char_file}")
        sys.exit(1)

    # Check if motion file exists (if provided)
    if args.motion_file and not os.path.exists(args.motion_file):
        print(f"Error: Motion file not found: {args.motion_file}")
        sys.exit(1)

    # Run environment
    try:
        run_environment(args)
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("\nNote: mjlab must be installed to run this script.")
        print("Install with: pip install mjlab")
        print("Or: uv add mjlab")
        sys.exit(1)


if __name__ == "__main__":
    main()
