"""
Example script for launching PARC motion tracking with mjlab.

Usage:
    uv run PARC/motion_tracker/envs/mj_launch_example.py \
        --char-file data/assets/humanoid.xml \
        --num-envs 4096

With motion tracking:
    uv run PARC/motion_tracker/envs/mj_launch_example.py \
        --char-file data/assets/humanoid.xml \
        --motion-file data/motion_terrains/civilization_motions.yaml \
        --num-envs 4096
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="PARC Motion Tracking with mjlab")
    parser.add_argument("--char-file", type=str, required=True, help="Path to character MJCF file")
    parser.add_argument("--motion-file", type=str, default=None, help="Path to motion library file")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--episode-length", type=float, default=10.0, help="Episode length in seconds")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate paths
    if not os.path.exists(args.char_file):
        print(f"Error: Character file not found: {args.char_file}")
        sys.exit(1)

    if args.motion_file and not os.path.exists(args.motion_file):
        print(f"Error: Motion file not found: {args.motion_file}")
        sys.exit(1)

    # Import mjlab (will fail if not installed)
    try:
        from mjlab.envs import ManagerBasedRlEnv
    except ImportError as e:
        print(f"Error: mjlab not installed: {e}")
        print("\nInstall with: uv add mjlab")
        sys.exit(1)

    # Import our config
    from parc.motion_tracker.envs.mj_env_config import make_parc_tracking_env_cfg

    print("=" * 60)
    print("PARC Motion Tracking with mjlab")
    print("=" * 60)
    print(f"Character file: {args.char_file}")
    print(f"Motion file: {args.motion_file or 'None'}")
    print(f"Num envs: {args.num_envs}")
    print(f"Episode length: {args.episode_length}s")
    print("=" * 60)

    # Create environment config
    cfg = make_parc_tracking_env_cfg(
        char_file=os.path.abspath(args.char_file),
        motion_file=os.path.abspath(args.motion_file) if args.motion_file else None,
        num_envs=args.num_envs,
        episode_length_s=args.episode_length,
    )

    # Create environment
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="human" if args.render else None)

    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs['policy'].shape}")

    # Run
    import torch
    total_reward = torch.zeros(args.num_envs, device=env.device)

    # Get action dimension (last dim of action space shape)
    action_dim = env.action_space.shape[-1]
    print(f"Action dimension: {action_dim}")

    print(f"\nRunning {args.num_steps} steps...")
    for step in range(args.num_steps):
        # Random actions
        action = torch.zeros(args.num_envs, action_dim, device=env.device)
        action += 0.1 * torch.randn_like(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.num_steps}, Mean reward: {total_reward.mean().item() / (step + 1):.4f}")

    print(f"\nDone. Mean total reward: {total_reward.mean().item():.4f}")
    env.close()


if __name__ == "__main__":
    main()
