"""
Standalone character observation utilities.

Extracted from ig_char_env.py to remove Isaac Gym dependency.
All functions are pure PyTorch with @torch.jit.script decorators.
"""

import torch

import parc.util.torch_util as torch_util


@torch.jit.script
def compute_char_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)

    if (global_obs):
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel
    else:
        local_root_rot = torch_util.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1], joint_rot.shape[2]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1] * joint_rot_obs_flat.shape[-1]])

    obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]

    if (len(key_pos) > 0):
        root_pos_expand = root_pos.unsqueeze(-2)
        key_pos = key_pos - root_pos_expand
        if (not global_obs):
            heading_rot_expand = heading_rot.unsqueeze(-2)
            heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
            flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                                    heading_rot_expand.shape[2])
            flat_body_pos = key_pos.reshape(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
            flat_local_body_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_body_pos)
            key_pos = flat_local_body_pos.reshape(key_pos.shape[0], key_pos.shape[1], key_pos.shape[2])

        key_pos_flat = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2]])
        obs = obs + [key_pos_flat]

    if (root_height_obs):
        root_h = root_pos[:, 2:3]
        obs = [root_h] + obs

    obs = torch.cat(obs, dim=-1)
    return obs
