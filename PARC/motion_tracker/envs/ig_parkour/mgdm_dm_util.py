from collections import OrderedDict
from typing import Optional

import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motion_tracker.envs.base_env as base_env
import parc.motion_tracker.envs.char_obs_util as char_obs_util
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util


class RefCharEnv:
    def __init__(self, config, num_envs, device, visualize, char_model: kin_char_model.KinCharModel):

        env_config = config["env"]
        self._num_envs = num_envs
        self._device = device
        self._visualize = visualize
        self._kin_char_model = char_model

        self._timestep = 1.0 / env_config["control_freq"]
        self.set_rand_root_pos_offset_scale(env_config["rand_root_pos_offset_scale"])

        self._root_pos_offset = None
        self._root_rot_offset = None
        self._root_vel_offset = None
        self._root_ang_vel_offset = None
        self._dof_pos_offset = None
        self._dof_vel_offset = None

        self._max_obs_h = env_config["max_obs_h"]
        self._min_obs_h = env_config["min_obs_h"]
        return
    
    def get_sim_tensor_views(self, 
                             ref_root_pos, ref_root_rot, ref_root_vel, ref_root_ang_vel,
                             ref_body_pos, ref_joint_rot, ref_dof_pos, ref_dof_vel, ref_contacts,
                             char_root_pos, char_root_rot, char_root_vel, char_root_ang_vel,
                             char_dof_pos, char_dof_vel, char_contact_forces,
                             char_rigid_body_pos, char_rigid_body_vel, char_rigid_body_ang_vel,
                             ):

        self._ref_root_pos = ref_root_pos
        self._ref_root_rot = ref_root_rot
        self._ref_root_vel = ref_root_vel
        self._ref_root_ang_vel = ref_root_ang_vel
        self._ref_body_pos = ref_body_pos
        self._ref_joint_rot = ref_joint_rot
        self._ref_dof_pos = ref_dof_pos
        self._ref_dof_vel = ref_dof_vel
        self._ref_contacts = ref_contacts

        self._char_root_pos = char_root_pos
        self._char_root_rot = char_root_rot
        self._char_root_vel = char_root_vel
        self._char_root_ang_vel = char_root_ang_vel
        self._char_dof_pos = char_dof_pos
        self._char_dof_vel = char_dof_vel
        self._char_contact_forces = char_contact_forces
        self._char_rigid_body_pos = char_rigid_body_pos
        self._char_rigid_body_vel = char_rigid_body_vel
        self._char_rigid_body_ang_vel = char_rigid_body_ang_vel
        return
    
    def get_data_buffer_views(self, reward_buf, done_buf, time_buf, timestep_buf, 
                              actors_need_reset,
                              env_offsets, key_body_ids, ray_xy_points, ray_hfs):
    
        self._reward_buf = reward_buf
        self._done_buf = done_buf
        self._time_buf = time_buf
        self._timestep_buf = timestep_buf
        self._actors_need_reset = actors_need_reset
        self._env_offsets = env_offsets
        self._key_body_ids = key_body_ids
        self._ray_xy_points = ray_xy_points
        self._ray_hfs = ray_hfs
        return

    def get_obs_buf_view(self, obs_buf):
        self._obs_buf = obs_buf
        return

    def _has_key_bodies(self):
        return len(self._key_body_ids) > 0
    
    def _char_state_init_from_ref(self, env_ids):
        self._char_rigid_body_vel[env_ids] = 0.0
        self._char_rigid_body_ang_vel[env_ids] = 0.0

        self._char_root_pos[env_ids] = self._ref_root_pos[env_ids] # note, actually a deepcopy.clone()
        self._char_root_rot[env_ids] = self._ref_root_rot[env_ids]
        self._char_root_vel[env_ids] = self._ref_root_vel[env_ids]
        self._char_root_ang_vel[env_ids] = self._ref_root_ang_vel[env_ids]
        
        self._char_dof_pos[env_ids] = self._ref_dof_pos[env_ids]
        self._char_dof_vel[env_ids] = self._ref_dof_vel[env_ids]
        return
    
    def add_noise_to_char_state(self, env_ids):
        root_pos_offset = torch.rand(size=[env_ids.shape[0], 2], device=self._device, dtype=torch.float32)*2.0 - 1.0
        root_pos_offset = self._rand_root_pos_offset_scale * root_pos_offset
        self._char_root_pos[env_ids, 0:2] += root_pos_offset
        return
    
    def apply_offsets_to_char_state(self, env_ids):
        if self._root_pos_offset is not None:
            self._char_root_pos[env_ids] += self._root_pos_offset[env_ids]

        if self._root_rot_offset is not None:
            self._char_root_rot[env_ids] = torch_util.quat_multiply(self._root_rot_offset[env_ids], self._char_root_rot[env_ids])

        if self._root_vel_offset is not None:
            self._char_root_vel[env_ids] += self._root_vel_offset
        
        if self._root_ang_vel_offset is not None:
            self._char_root_ang_vel[env_ids] += self._root_ang_vel_offset

        if self._dof_pos_offset is not None:
            self._char_dof_pos[env_ids] += self._dof_pos_offset

        if self._dof_vel_offset is not None:
            self._char_dof_vel[env_ids] += self._dof_vel_offset
        return
    
    def _refresh_ray_obs_hfs(self, char_root_pos_xyz, char_heading):

        char_root_pos_xy = char_root_pos_xyz[..., 0:2]

        num_ray_points = self._ray_xy_points.shape[0]
        ray_xy_points = self._ray_xy_points.unsqueeze(0).expand(self._num_envs, -1, -1)
        char_heading = char_heading.unsqueeze(-1).expand(-1, ray_xy_points.shape[1])
        ray_xy_points = torch_util.rotate_2d_vec(ray_xy_points, char_heading) + char_root_pos_xy.unsqueeze(1)
        ray_xy_points = ray_xy_points.view(ray_xy_points.shape[0] * ray_xy_points.shape[1], 2)
        ray_hfs = self._terrain.get_hf_val_from_points(ray_xy_points).view(self._num_envs, num_ray_points)

        # RELATIVE TO ROOT
        self._ray_hfs[...] = ray_hfs - char_root_pos_xyz[..., 2].unsqueeze(-1)
        self._ray_hfs[...] = torch.clamp(self._ray_hfs, min=self._min_obs_h, max=self._max_obs_h)

        if self._visualize:
            self._ray_xyz_points = torch.cat([ray_xy_points.view(self._num_envs, -1, 2), ray_hfs.unsqueeze(-1)], dim=-1)
        return

    def update_done(self, termination_height, episode_length, contact_body_ids, 
                    pose_termination, pose_termination_dist, global_obs, enable_early_termination,
                    track_root, root_pos_termination_dist, root_rot_termination_angle):
        global_body_pos = self._char_rigid_body_pos[..., 0:2] + self._env_offsets[:, 0:2].unsqueeze(1)
        body_grid_inds = self._terrain.get_grid_index(global_body_pos)
        termination_heights = self._terrain.hf[body_grid_inds[..., 0], body_grid_inds[..., 1]] + termination_height

        self._done_buf[:] = compute_done(done_buf=self._done_buf,
                                         time=self._time_buf, 
                                         ep_len=episode_length, 
                                         root_rot=self._char_root_rot,
                                         body_pos=self._char_rigid_body_pos,
                                         char_root_pos=self._char_root_pos,
                                         tar_root_rot=self._ref_root_rot,
                                         tar_body_pos=self._ref_body_pos,
                                         contact_force=self._char_contact_forces,
                                         contact_body_ids=contact_body_ids,
                                         termination_heights=termination_heights,
                                         pose_termination=pose_termination,
                                         pose_termination_dist=pose_termination_dist,
                                         global_obs=global_obs,
                                         enable_early_termination=enable_early_termination,
                                         track_root=track_root,
                                         root_pos_termination_dist=root_pos_termination_dist,
                                         root_rot_termination_angle=root_rot_termination_angle
                                         )
        
    def set_rand_root_pos_offset_scale(self, val):
        self._rand_root_pos_offset_scale = val    
        return
    
    def set_root_pos_offset(self, val: Optional[torch.Tensor] = None):
        if isinstance(val, torch.Tensor):
            assert val.shape[0] == self._num_envs
            assert val.shape[1] == 3
        self._root_pos_offset = val
        return
    
    def set_root_rot_offset(self, val: Optional[torch.Tensor] = None):
        if isinstance(val, torch.Tensor):
            assert val.shape[0] == self._num_envs
            assert val.shape[1] == 4
        self._root_rot_offset = val
        return
    
    def set_root_vel_offset(self, val: Optional[torch.Tensor] = None):
        if isinstance(val, torch.Tensor):
            assert val.shape[0] == self._num_envs
            assert val.shape[1] == 3
        self._root_vel_offset = val
        return
    
    def set_root_ang_vel_offset(self, val: Optional[torch.Tensor] = None):
        if isinstance(val, torch.Tensor):
            assert val.shape[0] == self._num_envs
            assert val.shape[1] == 3
        self._root_ang_vel_offset = val
        return
    
    def set_dof_pos_offset(self, val: Optional[torch.Tensor] = None):
        if isinstance(val, torch.Tensor):
            assert val.shape[0] == self._num_envs
            assert val.shape[1] == self._kin_char_model.get_dof_size()
        self._dof_pos_offset = val
        return
    
    def set_dof_vel_offset(self, val: Optional[torch.Tensor] = None):
        if isinstance(val, torch.Tensor):
            assert val.shape[0] == self._num_envs
            assert val.shape[1] == self._kin_char_model.get_dof_size()
        self._dof_vel_offset = val
        return
    
    
def fetch_tar_obs_data(motion_ids, 
                       motion_times, 
                       mlib: motion_lib.MotionLib,
                       timestep, 
                       tar_obs_steps):
    n = motion_ids.shape[0]
    num_steps = tar_obs_steps.shape[0]
    assert(num_steps > 0)
    
    motion_times = motion_times.unsqueeze(-1)
    time_steps = timestep * tar_obs_steps
    motion_times = motion_times + time_steps
    motion_ids_tiled = torch.broadcast_to(motion_ids.unsqueeze(-1), motion_times.shape)

    motion_ids_tiled = motion_ids_tiled.flatten()
    motion_times = motion_times.flatten()

    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = mlib.calc_motion_frame(motion_ids_tiled, motion_times)

    root_pos = root_pos.reshape([n, num_steps, root_pos.shape[-1]])
    root_rot = root_rot.reshape([n, num_steps, root_rot.shape[-1]])
    joint_rot = joint_rot.reshape([n, num_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
    contacts = contacts.reshape([n, num_steps, -1])
    return root_pos, root_rot, joint_rot, contacts

@torch.jit.script
def convert_to_local(root_rot, root_vel, root_ang_vel, key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    local_root_rot = torch_util.quat_mul(heading_inv_rot, root_rot)
    local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)
    local_root_ang_vel = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)
    
    if (len(key_pos) > 0):
        heading_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
        flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                                heading_rot_expand.shape[2])
        flat_key_pos = key_pos.reshape(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
        flat_local_key_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_key_pos)
        local_key_pos = flat_local_key_pos.reshape(key_pos.shape[0], key_pos.shape[1], key_pos.shape[2])
    else:
        local_key_pos = key_pos

    return local_root_rot, local_root_vel, local_root_ang_vel, local_key_pos

@torch.jit.script
def compute_deepmimic_reward(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, 
                             dof_vel, key_pos,
                             tar_root_pos, tar_root_rot, tar_root_vel, tar_root_ang_vel,
                             tar_joint_rot, tar_dof_vel, tar_key_pos,
                             joint_rot_err_w, dof_err_w, track_root_h,
                             track_root):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    pose_scale = 0.25
    vel_scale = 0.01
    root_pose_scale = 5.0
    root_vel_scale = 1.0
    key_pos_scale = 10.0
    
    pose_diff = torch_util.quat_diff_angle(joint_rot, tar_joint_rot)
    pose_errs = joint_rot_err_w * pose_diff * pose_diff
    pose_err = torch.sum(pose_errs, dim=-1)

    vel_diff = tar_dof_vel - dof_vel
    vel_errs = dof_err_w * vel_diff * vel_diff
    vel_err = torch.sum(vel_errs, dim=-1)

    root_pos_diff = tar_root_pos - root_pos

    if (not track_root):
        root_pos_diff[..., 0:2] = 0

    if (not track_root_h):
        root_pos_diff[..., 2] = 0

    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)
        tar_key_pos = tar_key_pos - tar_root_pos.unsqueeze(-2)

    if (not track_root):
        root_rot, root_vel, root_ang_vel, key_pos = convert_to_local(root_rot, root_vel, root_ang_vel, key_pos)
        tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos = convert_to_local(tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos)
        
    root_rot_err = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err *= root_rot_err

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)

    if (len(key_pos) > 0):
        key_pos_diff = tar_key_pos - key_pos
        key_pos_err = torch.sum(key_pos_diff * key_pos_diff, dim=-1)
        key_pos_err = torch.sum(key_pos_err, dim=-1)
    else:
        key_pos_err = torch.zeros([0], device=key_pos.device)

    pose_r = torch.exp(-pose_scale * pose_err)
    vel_r = torch.exp(-vel_scale * vel_err)
    root_pose_r = torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
    root_vel_r = torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))
    key_pos_r = torch.exp(-key_pos_scale * key_pos_err)

    return torch.stack([pose_r, vel_r, root_pose_r, root_vel_r, key_pos_r], dim=1)

@torch.jit.script
def compute_done(done_buf, time, ep_len, root_rot, body_pos, char_root_pos, tar_root_rot, tar_body_pos, 
                 contact_force, contact_body_ids, termination_heights,
                 pose_termination, pose_termination_dist, 
                 global_obs, enable_early_termination,
                 track_root, root_pos_termination_dist, root_rot_termination_angle):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool, bool, bool, float, float) -> Tensor
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)

    timeout = time >= ep_len
    done[timeout] = base_env.DoneFlags.TIME.value

    if (enable_early_termination):
        failed = torch.zeros(done.shape, device=done.device, dtype=torch.bool)

        if (contact_body_ids.shape[0] > 0):
            masked_contact_buf = contact_force.detach().clone()
            masked_contact_buf[:, contact_body_ids, :] = 0
            fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
            fall_contact = torch.any(fall_contact, dim=-1)

            body_height = body_pos[..., 2]
            fall_height = body_height < termination_heights
            fall_height[:, contact_body_ids] = False
            fall_height = torch.any(fall_height, dim=-1)

            has_fallen = torch.logical_and(fall_contact, fall_height)
            #print("has_fallen:", has_fallen[0].item())
            failed = torch.logical_or(failed, has_fallen)


        # head termination
        #head_pos = body_pos[:, 2, 2]
        #head_fail = head_pos < 0.2 # TODO make this a param
        #failed = torch.logical_or(failed, head_fail)

        if pose_termination:
            root_pos = body_pos[..., 0:1, :]
            tar_root_pos = tar_body_pos[..., 0:1, :]
            body_pos = body_pos[..., 1:, :] - root_pos
            tar_body_pos = tar_body_pos[..., 1:, :] - tar_root_pos

            #if (not global_obs):
            #    body_pos = ig_char_env.convert_to_local_body_pos(root_rot, body_pos)
            #    tar_body_pos = ig_char_env.convert_to_local_body_pos(tar_root_rot, tar_body_pos)

            body_pos_diff = tar_body_pos - body_pos
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)

            # body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
            pose_fail = torch.any(body_pos_dist > pose_termination_dist * pose_termination_dist, dim=-1)
            
            if (track_root):
                root_pos_diff = root_pos - tar_root_pos
                root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1).squeeze(-1)
                root_pos_fail = root_pos_dist > root_pos_termination_dist * root_pos_termination_dist
                root_rot_err = torch_util.quat_diff_angle(root_rot, tar_root_rot)
                root_rot_fail = torch.abs(root_rot_err) > root_rot_termination_angle
                pose_fail = torch.logical_or(pose_fail, root_pos_fail)
                pose_fail = torch.logical_or(pose_fail, root_rot_fail)

            failed = torch.logical_or(failed, pose_fail)

        # only fail after first timestep
        not_first_step = (time > 1e-5)
        failed = torch.logical_and(failed, not_first_step)
        done[failed] = base_env.DoneFlags.FAIL.value
    
    return done

@torch.jit.script
def compute_tar_obs(ref_root_pos, ref_root_rot, tar_root_pos, tar_root_rot, 
                    joint_rot, tar_key_pos,
                    global_obs, global_tar_root_h_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    ref_root_pos = ref_root_pos.unsqueeze(-2)
    root_pos_obs = tar_root_pos - ref_root_pos
    #print("root_pos_obs:", root_pos_obs[0])
    #print("tar_root_pos")
    
    if (len(tar_key_pos) > 0):
        tar_key_pos = tar_key_pos - tar_root_pos.unsqueeze(-2)

    if (not global_obs): # target obs are relative to root rot
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, tar_root_pos.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                               heading_inv_rot_expand.shape[2]))
        root_pos_obs_flat = torch.reshape(root_pos_obs, [root_pos_obs.shape[0] * root_pos_obs.shape[1], root_pos_obs.shape[2]])
        root_pos_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
        root_pos_obs = torch.reshape(root_pos_obs_flat, tar_root_pos.shape)
        
        tar_root_rot = torch_util.quat_mul(heading_inv_rot_expand, tar_root_rot)

        if (len(tar_key_pos) > 0):
            heading_inv_rot_expand = heading_inv_rot_expand.unsqueeze(-2)
            heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 1, tar_key_pos.shape[2], 1))
            heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1] * heading_inv_rot_expand.shape[2],
                                                                   heading_inv_rot_expand.shape[3]))
            key_pos_flat = tar_key_pos.reshape((tar_key_pos.shape[0] * tar_key_pos.shape[1] * tar_key_pos.shape[2],
                                            tar_key_pos.shape[3]))
            key_pos_flat = torch_util.quat_rotate(heading_inv_rot_flat, key_pos_flat)
            tar_key_pos = key_pos_flat.reshape(tar_key_pos.shape)
            tar_key_pos = tar_key_pos + root_pos_obs.unsqueeze(2)

    if (global_tar_root_h_obs):
        root_pos_obs[..., 2] = tar_root_pos[..., 2]
    # else:
    #     # do nothing
    #     root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = torch.reshape(tar_root_rot, [tar_root_rot.shape[0] * tar_root_rot.shape[1], tar_root_rot.shape[2]])
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(root_rot_obs_flat, [tar_root_rot.shape[0], tar_root_rot.shape[1], root_rot_obs_flat.shape[-1]])

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]])
    
    obs = [root_pos_obs, root_rot_obs, joint_rot_obs]
    if (len(tar_key_pos) > 0):
        tar_key_pos = torch.reshape(tar_key_pos, [tar_key_pos.shape[0], tar_key_pos.shape[1], tar_key_pos.shape[2] * tar_key_pos.shape[3]])
        obs.append(tar_key_pos)

    obs = torch.cat(obs, dim=-1)

    return obs

def compute_deepmimic_obs(root_pos, root_rot, root_vel, root_ang_vel, 
                          joint_rot, dof_vel, key_pos, 
                          global_obs, root_height_obs, 
                          enable_tar_obs, tar_root_pos, tar_root_rot,
                          tar_joint_rot, tar_key_pos):
    ## type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
    char_obs = char_obs_util.compute_char_obs(root_pos=root_pos,
                                            root_rot=root_rot,
                                            root_vel=root_vel,
                                            root_ang_vel=root_ang_vel,
                                            joint_rot=joint_rot,
                                            dof_vel=dof_vel,
                                            key_pos=key_pos,
                                            global_obs=global_obs,
                                            root_height_obs=root_height_obs)

    obs = OrderedDict()
    obs["char_obs"] = char_obs

    if (enable_tar_obs):

        tar_obs = compute_tar_obs(ref_root_pos=root_pos,
                                  ref_root_rot=root_rot,
                                  tar_root_pos=tar_root_pos, 
                                  tar_root_rot=tar_root_rot, 
                                  joint_rot=tar_joint_rot,
                                  tar_key_pos=tar_key_pos,
                                  global_obs=global_obs,
                                  global_tar_root_h_obs=False)

        obs["tar_obs"] = tar_obs
    
    return obs

@torch.jit.script
def compute_contact_reward(tar_contacts, contact_forces, contact_weights):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    # check left and right foot contact forces using contact force tensor
    # Multiply by -(1-foot_contact weight) and add to reward.
    # This penalizes contacts when there shouldn't be contacts.

    forces = torch.norm(contact_forces, dim=2)

    # clamp so this penalty is bounded
    forces = torch.clamp_max(forces, 1.0)

    # penalty when contacting and not supposed to
    contact_r = -(1.0 - tar_contacts) * forces

    # reward when contacting and supposed to
    contact_r += tar_contacts * forces

    r = contact_weights * contact_r

    return r

@torch.jit.script
def compute_tracking_error(root_pos, root_rot, body_rot, body_pos,
                            tar_root_pos, tar_root_rot,
                            tar_body_rot, tar_body_pos,
                            root_vel, root_ang_vel, dof_vel,
                            tar_root_vel, tar_root_ang_vel, tar_dof_vel):

    pose_diff = torch_util.quat_diff_angle(body_rot, tar_body_rot)
    pose_err = torch.abs(pose_diff)
    pose_err = torch.mean(pose_err, dim=-1)

    root_pos_diff = tar_root_pos - root_pos
    root_pos_diff_l2 = torch.linalg.vector_norm(root_pos_diff, dim=-1)
    
    body_pos = body_pos - root_pos.unsqueeze(-2)
    tar_body_pos = tar_body_pos - tar_root_pos.unsqueeze(-2)
    body_pos_diff = tar_body_pos - body_pos
    body_pos_diff_l2 = torch.linalg.vector_norm(body_pos_diff, dim=-1)
    body_pos_err = torch.mean(body_pos_diff_l2, dim=-1)

    root_rot_diff = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err = torch.abs(root_rot_diff)

    dof_vel_diff = tar_dof_vel - dof_vel
    dof_vel_err = torch.mean(torch.abs(dof_vel_diff), dim=-1)

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.mean(torch.abs(root_vel_diff), dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.mean(torch.abs(root_ang_vel_diff), dim=-1)

    tracking_error = torch.stack([root_pos_diff_l2, root_rot_err, body_pos_err, pose_err, dof_vel_err, root_vel_err, root_ang_vel_err], dim=-1)
    return tracking_error