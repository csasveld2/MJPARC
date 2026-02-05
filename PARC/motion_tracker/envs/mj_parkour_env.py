"""
MjParkourEnv: mjlab-based motion tracking environment for PARC.

This replaces IGParkourEnv by using mjlab (MuJoCo Warp) as the physics backend
while maintaining the same BaseEnv interface expected by PARC's training pipeline.

Architecture:
- Uses ManagerBasedRlEnv internally for physics simulation
- Maintains standalone tensor buffers for character state (not entity.data properties)
- After each physics step, copies entity.data → buffers (_sync_from_sim)
- During reset, copies buffers → sim state (_write_reset_state)
- Delegates motion tracking to dm_env.DeepMimicEnv (pure PyTorch)
- Delegates reward/obs/termination to mgdm_dm_util (pure PyTorch)
"""

import os
import pickle
import time
from collections import OrderedDict

import gym
import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.motion_tracker.envs.base_env as base_env
import parc.motion_tracker.envs.ig_parkour.dm_env as dm_env
import parc.motion_tracker.envs.ig_parkour.mgdm_dm_util as mgdm_dm_util
import parc.motion_tracker.envs.char_obs_util as char_obs_util
import parc.util.geom_util as geom_util
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util
from parc.util.logger import Logger

# mjlab imports
import mujoco
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.viewer import ViewerConfig

SIM_CHAR_IDX = 0


class MjParkourEnv(base_env.BaseEnv):
    """Motion tracking environment using mjlab (MuJoCo Warp) as physics backend."""

    NAME = "mj_parkour"

    def get_reward_bounds(self):
        return (0.0, 1.0)

    def __init__(self, config, num_envs, device, visualize):
        super().__init__(visualize)

        self._start_compute_time = time.time()
        self._config = config
        env_config = config["env"]

        self._num_envs = num_envs
        self._device = device
        self._visualize = visualize

        self._bypass_record_fail = False
        self._fraction_dm_envs = env_config["fraction_dm_envs"]
        self._num_dm_envs = min(int(self._fraction_dm_envs * num_envs), num_envs)
        self._never_done = env_config.get("never_done", False)
        self._contact_detection_eps = env_config.get("contact_detection_eps", 1e-5)

        # Simulation parameters
        self._sim_freq = env_config["sim_freq"]
        self._control_freq = env_config["control_freq"]
        self._sim_steps_per_control = self._sim_freq // self._control_freq
        self._timestep = 1.0 / self._control_freq
        self._episode_length = env_config["episode_length"]
        self._global_obs = env_config.get("global_obs", False)

        # DeepMimic parameters
        self._enable_early_termination = env_config["enable_early_termination"]
        self._termination_height = torch.tensor(env_config["termination_height"], dtype=torch.float32, device=device)
        self._pose_termination = env_config.get("pose_termination", False)
        self._pose_termination_dist = torch.tensor(env_config["pose_termination_dist"], dtype=torch.float32, device=device)
        self._tar_obs_steps = env_config.get("tar_obs_steps", [1])
        self._tar_obs_steps = torch.tensor(self._tar_obs_steps, device=device, dtype=torch.int)
        self._rand_reset = env_config.get("rand_reset", True)
        self._enable_tar_obs = env_config.get("enable_tar_obs", True)
        self._global_root_height_obs = env_config["global_root_height_obs"]
        self._track_root = env_config["track_root"]
        self._track_root_h = env_config["track_root_h"]
        self._root_pos_termination_dist = env_config["root_pos_termination_dist"]
        self._root_rot_termination_angle = env_config["root_rot_termination_angle"]

        # Contact info
        self._use_contact_info = env_config["use_contact_info"]
        if self._use_contact_info:
            self._contact_weights = torch.tensor(env_config["contact_weights"], dtype=torch.float32, device=device)

        # Reward weights (normalized)
        self._pose_w = env_config["pose_w"]
        self._vel_w = env_config["vel_w"]
        self._root_pos_w = env_config["root_pos_w"]
        self._root_vel_w = env_config["root_vel_w"]
        self._key_pos_w = env_config["key_pos_w"]
        total_w = self._pose_w + self._vel_w + self._root_pos_w + self._root_vel_w + self._key_pos_w
        self._pose_w /= total_w
        self._vel_w /= total_w
        self._root_pos_w /= total_w
        self._root_vel_w /= total_w
        self._key_pos_w /= total_w

        self._report_tracking_error = env_config.get("report_tracking_error", False)
        self._output_motion_dir = env_config.get("output_motion_dir", "output/_motions/recorded_motions/")
        os.makedirs(self._output_motion_dir, exist_ok=True)

        # Heightmap rays
        self._use_heightmap = True
        ray_points_behind = env_config["ray_points_behind"]
        ray_points_ahead = env_config["ray_points_ahead"]
        ray_num_left = env_config["ray_num_left"]
        ray_num_right = env_config["ray_num_right"]
        ray_dx = env_config["ray_dx"]
        ray_angle = env_config["ray_angle"]
        self._ray_xy_points = geom_util.get_xy_points_cone(
            center=torch.zeros(size=(2,), dtype=torch.float32, device=device),
            dx=ray_dx, num_neg=ray_points_behind, num_pos=ray_points_ahead,
            num_rays_neg=ray_num_left, num_rays_pos=ray_num_right,
            angle_between_rays=ray_angle)
        num_points = self._ray_xy_points.shape[0]
        self._ray_hfs = torch.zeros(size=(num_envs, num_points), dtype=torch.float32, device=device)
        self._max_obs_h = env_config["max_obs_h"]
        self._min_obs_h = env_config["min_obs_h"]

        # Build kinematic character model (same pattern as original ig_char_env)
        char_file = env_config["char_file"]
        self._build_kin_char_model(char_file)

        # Key bodies and contact bodies
        key_body_names = env_config.get("key_bodies", [])
        self._key_body_ids = self._build_body_ids_tensor(key_body_names)
        contact_bodies = env_config.get("contact_bodies", [])
        self._contact_body_ids = self._build_body_ids_tensor(contact_bodies)

        # Joint error weights
        joint_err_w = env_config.get("joint_err_w", None)
        self._parse_joint_err_weights(joint_err_w)

        # Build DeepMimic environment (motion tracking, pure PyTorch)
        if self.has_dm_envs():
            self._dm_env = dm_env.DeepMimicEnv(config, self._num_dm_envs, device, visualize, self._kin_char_model)
            self._init_dm_terrain(env_config)

        # Build mjlab simulation
        self._build_mjlab_env(char_file, env_config)

        # Build all tensor buffers and connect them to dm_env
        self._build_sim_tensors(env_config)
        self._build_data_buffers()

        # Build action space (PD position targets)
        self._action_space = self._build_action_space()

        self.set_write_agent_states_flag(env_config.get("write_agent_states", False))
        if self.is_writing_agent_states():
            self.build_agent_states_dict()

        self._demo_mode = env_config["demo_mode"]
        Logger.print("MjParkourEnv initialized with {} envs on device {}".format(num_envs, device))

    # ==================== Initialization ====================

    def _build_kin_char_model(self, char_file):
        """Build kinematic character model from XML file."""
        _, file_ext = os.path.splitext(char_file)
        assert file_ext == ".xml", "Unsupported character file format: {}".format(file_ext)
        char_model = kin_char_model.KinCharModel(self._device)
        char_model.load_char_file(char_file)
        self._kin_char_model = char_model

    def _build_body_ids_tensor(self, body_names):
        """Convert body names to index tensor using kin_char_model."""
        if not body_names:
            return torch.tensor([], dtype=torch.long, device=self._device)
        body_ids = []
        all_body_names = self._kin_char_model.get_body_names()
        for name in body_names:
            if name in all_body_names:
                idx = all_body_names.index(name)
                body_ids.append(idx)
            else:
                Logger.print("Warning: body '{}' not found in character model".format(name))
        return torch.tensor(body_ids, dtype=torch.long, device=self._device)

    def _init_dm_terrain(self, env_config):
        """Initialize terrain data for DeepMimic env.

        In mjlab, the physical terrain is a flat plane provided by TerrainImporterCfg.
        We initialize dm_env's terrain attributes directly:
        - _dm_motion_offsets: zeros (no terrain offsets on a flat plane)
        - _terrain: a flat SubTerrain for heightmap queries (returns 0 everywhere)
        """
        dm_config = env_config["dm"]
        dm_terrain_save_path = dm_config.get("terrain_save_path", None)

        # Try to load pre-built terrain if it exists
        if dm_terrain_save_path is not None and os.path.exists(dm_terrain_save_path):
            self._dm_env.load_terrain(dm_terrain_save_path)
            Logger.print("Loaded pre-built terrain from {}".format(dm_terrain_save_path))
            return

        # Check if all motions have terrain data
        has_all_terrains = all(t is not None for t in self._dm_env._mlib._terrains)

        if has_all_terrains:
            # Use dm_env's built-in terrain building (generates mesh verts we discard)
            self._dm_env.build_terrain(env_config, dm_terrain_save_path)
            Logger.print("Built DeepMimic terrain from motion data")
        else:
            # Create a flat terrain — used for heightmap queries and motion offset tracking
            env_spacing = env_config.get("env_spacing", 5.0)
            terrain_size = max(env_spacing * 4 * int(np.sqrt(self._num_envs)), 100.0)
            dx = dm_config.get("heightmap", {}).get("horizontal_scale", 0.4)
            num_cells = int(terrain_size / dx)

            self._dm_env._terrain = terrain_util.SubTerrain(
                terrain_name="flat_plane",
                x_dim=num_cells, y_dim=num_cells,
                dx=dx, dy=dx,
                min_x=-terrain_size / 2.0, min_y=-terrain_size / 2.0,
                device=self._device,
            )

            num_motions = self._dm_env._mlib.num_motions()
            terrains_per_motion = self._dm_env._terrains_per_motion
            self._dm_env._dm_motion_offsets = torch.zeros(
                size=[num_motions, terrains_per_motion, 2],
                dtype=torch.float32, device=self._device,
            )
            Logger.print("Initialized flat terrain for mjlab ({}x{} cells)".format(num_cells, num_cells))

    def _build_mjlab_env(self, char_file, env_config):
        """Build the mjlab simulation environment."""
        char_file_abs = os.path.abspath(char_file)

        def load_humanoid_spec():
            return mujoco.MjSpec.from_file(char_file_abs)

        # Configure articulation with PD actuators matching humanoid joint groups
        humanoid_articulation = EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=("abdomen_.*",),
                    stiffness=1000.0, damping=100.0, effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("neck_.*",),
                    stiffness=100.0, damping=10.0, effort_limit=50.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_shoulder_.*",),
                    stiffness=400.0, damping=40.0, effort_limit=100.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_elbow",),
                    stiffness=300.0, damping=30.0, effort_limit=70.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_.*",),
                    stiffness=500.0, damping=50.0, effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_knee",),
                    stiffness=500.0, damping=50.0, effort_limit=150.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_ankle_.*",),
                    stiffness=400.0, damping=40.0, effort_limit=90.0,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        )

        init_pos = (0.0, 0.0, 0.882416)  # default standing height

        # Minimal mjlab config — we handle obs/rewards/termination ourselves
        obs_terms = {
            "dummy": ObservationTermCfg(func=mjlab_mdp.base_lin_vel),
        }

        physics_dt = 1.0 / self._sim_freq

        cfg = ManagerBasedRlEnvCfg(
            scene=SceneCfg(
                num_envs=self._num_envs,
                env_spacing=env_config.get("env_spacing", 5.0),
                terrain=TerrainImporterCfg(terrain_type="plane"),
                entities={
                    "robot": EntityCfg(
                        spec_fn=load_humanoid_spec,
                        init_state=EntityCfg.InitialStateCfg(
                            pos=init_pos,
                            rot=(1.0, 0.0, 0.0, 0.0),
                        ),
                        articulation=humanoid_articulation,
                    ),
                },
            ),
            sim=SimulationCfg(
                nconmax=200,
                njmax=600,
                mujoco=MujocoCfg(
                    timestep=physics_dt,
                    solver="newton",
                    iterations=4,
                    ls_iterations=4,
                ),
            ),
            viewer=ViewerConfig(
                lookat=(0.0, 0.0, 1.0),
                distance=5.0,
                elevation=-20.0,
                azimuth=90.0,
            ),
            observations={"policy": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True)},
            actions={"joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=1.0,
                use_default_offset=False,
            )},
            commands={},
            events={"reset_scene": EventTermCfg(func=mjlab_mdp.reset_scene_to_default, mode="reset")},
            rewards={"is_alive": RewardTermCfg(func=mjlab_mdp.is_alive, weight=1.0)},
            terminations={"time_out": TerminationTermCfg(func=mjlab_mdp.time_out, time_out=True)},
            episode_length_s=self._episode_length,
            decimation=self._sim_steps_per_control,
        )

        self._mjlab_env = ManagerBasedRlEnv(cfg=cfg, device=self._device)
        self._robot = self._mjlab_env.scene.entities["robot"]

        # Build env offsets for grid layout
        env_spacing = env_config.get("env_spacing", 5.0)
        num_env_per_row = int(np.sqrt(self._num_envs))
        self._env_offsets = torch.zeros(size=(self._num_envs, 3), dtype=torch.float32, device=self._device)
        for i in range(self._num_envs):
            curr_col = i % num_env_per_row
            curr_row = i // num_env_per_row
            self._env_offsets[i, 0] = env_spacing * 2 * curr_col
            self._env_offsets[i, 1] = env_spacing * 2 * curr_row

        # Find robot body offset in MuJoCo model
        self._robot_body_offset = self._find_robot_body_offset()

    def _find_robot_body_offset(self):
        """Find the offset of robot bodies in the full MuJoCo model's cfrc_ext."""
        robot_body_names = self._robot.body_names
        mj_model = self._mjlab_env.sim.mj_model
        for i in range(mj_model.nbody):
            name = mj_model.body(i).name
            clean_name = name.split("/")[-1] if "/" in name else name
            if clean_name == robot_body_names[0]:
                return i
        Logger.print("Warning: Could not find robot body offset, using 0")
        return 0

    def _build_sim_tensors(self, env_config):
        """Allocate standalone tensor buffers for character state.

        These buffers are:
        - Written to by dm_env during reset (via tensor view slices)
        - Synced FROM simulation after each physics step (_sync_from_sim)
        - Synced TO simulation during reset (_write_reset_state)
        """
        num_envs = self._num_envs
        num_bodies = self._kin_char_model.get_num_joints()  # num joints == num bodies in PARC
        dof_size = self._kin_char_model.get_dof_size()
        num_joints_minus_root = num_bodies - 1
        d = self._device

        # Character state buffers (mutable — dm_env writes to these during reset)
        self._char_root_pos = torch.zeros(num_envs, 3, device=d, dtype=torch.float32)
        self._char_root_rot = torch.zeros(num_envs, 4, device=d, dtype=torch.float32)
        self._char_root_rot[:, 0] = 1.0  # identity quaternion (w,x,y,z)
        self._char_root_vel = torch.zeros(num_envs, 3, device=d, dtype=torch.float32)
        self._char_root_ang_vel = torch.zeros(num_envs, 3, device=d, dtype=torch.float32)

        self._char_dof_pos = torch.zeros(num_envs, dof_size, device=d, dtype=torch.float32)
        self._char_dof_vel = torch.zeros(num_envs, dof_size, device=d, dtype=torch.float32)

        self._char_rigid_body_pos = torch.zeros(num_envs, num_bodies, 3, device=d, dtype=torch.float32)
        self._char_rigid_body_rot = torch.zeros(num_envs, num_bodies, 4, device=d, dtype=torch.float32)
        self._char_rigid_body_rot[:, :, 0] = 1.0
        self._char_rigid_body_vel = torch.zeros(num_envs, num_bodies, 3, device=d, dtype=torch.float32)
        self._char_rigid_body_ang_vel = torch.zeros(num_envs, num_bodies, 3, device=d, dtype=torch.float32)

        self._char_contact_forces = torch.zeros(num_envs, num_bodies, 3, device=d, dtype=torch.float32)

        # Reference state buffers (populated by DeepMimicEnv)
        self._ref_root_pos = torch.zeros(num_envs, 3, device=d, dtype=torch.float32)
        self._ref_root_rot = torch.zeros(num_envs, 4, device=d, dtype=torch.float32)
        self._ref_root_vel = torch.zeros(num_envs, 3, device=d, dtype=torch.float32)
        self._ref_root_ang_vel = torch.zeros(num_envs, 3, device=d, dtype=torch.float32)
        self._ref_body_pos = torch.zeros(num_envs, num_bodies, 3, device=d, dtype=torch.float32)
        self._ref_joint_rot = torch.zeros(num_envs, num_joints_minus_root, 4, device=d, dtype=torch.float32)
        self._ref_dof_pos = torch.zeros(num_envs, dof_size, device=d, dtype=torch.float32)
        self._ref_dof_vel = torch.zeros(num_envs, dof_size, device=d, dtype=torch.float32)

        num_contact_rbs = self._kin_char_model.get_num_contact_bodies()
        if self._use_contact_info:
            self._ref_contacts = torch.zeros(num_envs, num_contact_rbs, device=d, dtype=torch.float32)

        # Need-reset buffer (1 actor per env — no ref char in mjlab)
        self._actors_need_reset = torch.zeros(num_envs, 1, device=d, dtype=torch.bool)

        # Pass tensor views to DeepMimic env
        if self.has_dm_envs():
            self._give_sim_tensor_views()

        # Initialize buffers from simulation state
        self._sync_from_sim()

    def _build_data_buffers(self):
        """Build reward/done/time/obs buffers expected by the training pipeline."""
        num_envs = self._num_envs

        self._reward_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._done_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._timestep_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._time_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._ep_num_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int64)
        self._info = dict()

        # Give data buffer views to DeepMimic env
        if self.has_dm_envs():
            self._give_data_buffer_views()

        # Build observation buffer
        obs_space = self.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_buf = torch.zeros([num_envs] + list(obs_space.shape), device=self._device, dtype=obs_dtype)

    def _build_action_space(self):
        """Build action space based on DOF limits."""
        dof_size = self._kin_char_model.get_dof_size()
        num_joints = self._kin_char_model.get_num_joints()

        low = np.zeros(dof_size, dtype=np.float32)
        high = np.zeros(dof_size, dtype=np.float32)

        for j in range(1, num_joints):
            curr_joint = self._kin_char_model.get_joint(j)
            j_dof_dim = curr_joint.get_dof_dim()
            if j_dof_dim > 0:
                if j_dof_dim == 3:  # spherical joint
                    limits = curr_joint.limits
                    if limits is not None:
                        j_low = limits[0].cpu().numpy()
                        j_high = limits[1].cpu().numpy()
                        curr_scale = max(np.max(np.abs(j_low)), np.max(np.abs(j_high)))
                        curr_scale = 1.2 * curr_scale
                    else:
                        curr_scale = np.pi
                    curr_low = -curr_scale * np.ones(j_dof_dim)
                    curr_high = curr_scale * np.ones(j_dof_dim)
                else:  # hinge joint
                    limits = curr_joint.limits
                    if limits is not None:
                        j_low = limits[0].cpu().numpy()
                        j_high = limits[1].cpu().numpy()
                        curr_mid = 0.5 * (j_high + j_low)
                        curr_scale = 0.7 * (j_high - j_low)
                        curr_low = curr_mid - curr_scale
                        curr_high = curr_mid + curr_scale
                    else:
                        curr_low = -np.pi * np.ones(j_dof_dim)
                        curr_high = np.pi * np.ones(j_dof_dim)

                dof_idx = curr_joint.dof_idx
                low[dof_idx:dof_idx + j_dof_dim] = curr_low
                high[dof_idx:dof_idx + j_dof_dim] = curr_high

        return gym.spaces.Box(low=low, high=high)

    def _give_sim_tensor_views(self):
        """Pass mutable tensor buffer slices to DeepMimic environment.

        dm_env stores these references and writes to them during reset().
        Since these are slices of our own buffers, the writes are visible to us.
        """
        self._dm_env.get_sim_tensor_views(
            ref_root_pos=self._get_dm_slice(self._ref_root_pos),
            ref_root_rot=self._get_dm_slice(self._ref_root_rot),
            ref_root_vel=self._get_dm_slice(self._ref_root_vel),
            ref_root_ang_vel=self._get_dm_slice(self._ref_root_ang_vel),
            ref_body_pos=self._get_dm_slice(self._ref_body_pos),
            ref_joint_rot=self._get_dm_slice(self._ref_joint_rot),
            ref_dof_pos=self._get_dm_slice(self._ref_dof_pos),
            ref_dof_vel=self._get_dm_slice(self._ref_dof_vel),
            ref_contacts=self._get_dm_slice(self._ref_contacts),
            char_root_pos=self._get_dm_slice(self._char_root_pos),
            char_root_rot=self._get_dm_slice(self._char_root_rot),
            char_root_vel=self._get_dm_slice(self._char_root_vel),
            char_root_ang_vel=self._get_dm_slice(self._char_root_ang_vel),
            char_dof_pos=self._get_dm_slice(self._char_dof_pos),
            char_dof_vel=self._get_dm_slice(self._char_dof_vel),
            char_contact_forces=self._get_dm_slice(self._char_contact_forces),
            char_rigid_body_pos=self._get_dm_slice(self._char_rigid_body_pos),
            char_rigid_body_vel=self._get_dm_slice(self._char_rigid_body_vel),
            char_rigid_body_ang_vel=self._get_dm_slice(self._char_rigid_body_ang_vel),
        )

    def _give_data_buffer_views(self):
        """Pass data buffer slices to DeepMimic environment."""
        self._dm_env.get_data_buffer_views(
            reward_buf=self._get_dm_slice(self._reward_buf),
            done_buf=self._get_dm_slice(self._done_buf),
            time_buf=self._get_dm_slice(self._time_buf),
            timestep_buf=self._get_dm_slice(self._timestep_buf),
            actors_need_reset=self._get_dm_slice(self._actors_need_reset),
            env_offsets=self._get_dm_slice(self._env_offsets),
            key_body_ids=self._key_body_ids,
            ray_xy_points=self._ray_xy_points,
            ray_hfs=self._get_dm_slice(self._ray_hfs),
        )

    # ==================== Simulation Sync ====================

    def _sync_from_sim(self):
        """Copy entity.data from mjlab simulation into our standalone buffers.

        Called after each physics step so that our buffers reflect the current sim state.
        """
        robot_data = self._robot.data

        self._char_root_pos[:] = robot_data.root_link_pos_w
        self._char_root_rot[:] = robot_data.root_link_quat_w
        self._char_root_vel[:] = robot_data.root_link_lin_vel_w
        self._char_root_ang_vel[:] = robot_data.root_link_ang_vel_w

        self._char_dof_pos[:] = robot_data.joint_pos
        self._char_dof_vel[:] = robot_data.joint_vel

        self._char_rigid_body_pos[:] = robot_data.body_link_pos_w
        self._char_rigid_body_rot[:] = robot_data.body_link_quat_w
        self._char_rigid_body_vel[:] = robot_data.body_link_lin_vel_w
        self._char_rigid_body_ang_vel[:] = robot_data.body_link_ang_vel_w

        # Contact forces from MuJoCo cfrc_ext
        cfrc_ext = self._mjlab_env.sim.data.cfrc_ext
        num_robot_bodies = self._robot.num_bodies
        body_offset = self._robot_body_offset
        # cfrc_ext shape: (num_envs, total_bodies, 6) — last 3 are forces
        self._char_contact_forces[:, :num_robot_bodies, :] = \
            cfrc_ext[:, body_offset:body_offset + num_robot_bodies, 3:6]

    def _detect_nan_envs(self):
        """Detect environments with NaN in physics state and sanitize them.

        Returns tensor of env indices that had NaN. Zeros out their state buffers
        so downstream obs/reward computations don't propagate NaN.
        """
        nan_mask = (
            torch.isnan(self._char_root_pos).any(dim=-1) |
            torch.isnan(self._char_root_rot).any(dim=-1) |
            torch.isnan(self._char_dof_pos).any(dim=-1)
        )
        nan_envs = nan_mask.nonzero(as_tuple=False).flatten()
        if len(nan_envs) > 0:
            # Zero out NaN state so obs/reward don't produce NaN
            self._char_root_pos[nan_envs] = 0.0
            self._char_root_pos[nan_envs, 2] = 0.5  # above ground
            self._char_root_rot[nan_envs] = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self._device)
            self._char_root_vel[nan_envs] = 0.0
            self._char_root_ang_vel[nan_envs] = 0.0
            self._char_dof_pos[nan_envs] = 0.0
            self._char_dof_vel[nan_envs] = 0.0
            self._char_rigid_body_pos[nan_envs] = 0.0
            self._char_rigid_body_rot[nan_envs] = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self._device)
            self._char_rigid_body_vel[nan_envs] = 0.0
            self._char_rigid_body_ang_vel[nan_envs] = 0.0
            self._char_contact_forces[nan_envs] = 0.0
        return nan_envs

    def _write_reset_state(self, env_ids):
        """Write character state from our buffers into the mjlab simulation.

        Called during reset after dm_env has written the desired state into our buffers.
        Only writes for the specified env_ids.
        """
        if len(env_ids) == 0:
            return

        # Build root state tensor: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
        # Index by env_ids so shape matches what mjlab expects
        root_state = torch.cat([
            self._char_root_pos[env_ids],
            self._char_root_rot[env_ids],
            self._char_root_vel[env_ids],
            self._char_root_ang_vel[env_ids],
        ], dim=-1)

        self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(
            self._char_dof_pos[env_ids], self._char_dof_vel[env_ids], env_ids=env_ids
        )

    # ==================== BaseEnv Interface ====================

    def get_num_envs(self):
        return self._num_envs

    def get_obs_space(self):
        """Compute observation space by running _compute_obs on current state.

        Overrides base_env.get_obs_space() which calls reset().
        We compute obs directly to avoid chicken-and-egg with _obs_buf.
        """
        obs = self._compute_obs()
        obs_shape = list(obs.shape[1:])
        obs_dtype = torch_util.torch_dtype_to_numpy(obs.dtype)
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_shape, dtype=obs_dtype,
        )
        return obs_space

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.long)

        if len(env_ids) == 0:
            return self._obs_buf, self._info

        # dm_env.reset() writes reference state + char state into our buffers
        if self.has_dm_envs():
            dm_env_ids = self._extract_dm_env_ids(env_ids)
            self._dm_env.reset(dm_env_ids)

        # Push the char state from our buffers into the mjlab simulation
        self._write_reset_state(env_ids)

        # Step sim once to settle, then sync back
        self._mjlab_env.sim.step()
        self._mjlab_env.scene.update(self._mjlab_env.sim)
        self._sync_from_sim()

        # Detect and sanitize NaN from physics
        self._detect_nan_envs()

        # Reset time/done
        self._timestep_buf[env_ids] = 0
        self._time_buf[env_ids] = 0.0
        self._done_buf[env_ids] = base_env.DoneFlags.NULL.value
        self._ep_num_buf[env_ids] += 1

        # Refresh heightmap observations
        self._refresh_obs_hfs()

        # Compute observations
        self._update_observations(env_ids)
        self._update_info(env_ids)

        # Sanitize any remaining NaN in obs
        obs_nan_mask = torch.isnan(self._obs_buf).any(dim=-1)
        if obs_nan_mask.any():
            self._obs_buf[obs_nan_mask] = 0.0

        return self._obs_buf, self._info

    def step(self, action):
        self._start_compute_time = time.time()

        # Apply actions
        self._pre_physics_step(action)

        # Step physics (multiple substeps for decimation)
        for _ in range(self._sim_steps_per_control):
            self._mjlab_env.sim.step()
        self._mjlab_env.scene.update(self._mjlab_env.sim)

        # Sync state from simulation into our buffers
        self._sync_from_sim()

        # Detect NaN from physics divergence and mark those envs as failed
        nan_envs = self._detect_nan_envs()

        # Update time
        self._time_buf += self._timestep
        self._timestep_buf += 1

        # Update reference motion
        if self.has_dm_envs():
            self._dm_env._update_ref_motion()

        # Refresh heightmap observations
        self._refresh_obs_hfs()

        # Compute reward
        self._update_reward()

        # Compute done
        self._update_done()

        # Force NaN envs to fail (after _update_done so they get reset)
        if len(nan_envs) > 0:
            self._done_buf[nan_envs] = base_env.DoneFlags.FAIL.value
            self._reward_buf[nan_envs] = 0.0

        if self._never_done:
            self._done_buf[:] = base_env.DoneFlags.NULL.value

        # Write agent states if recording
        self.write_agent_states()

        # Compute observations for ALL envs (matching original IGParkourEnv)
        self._update_observations()
        self._update_info()

        # Final NaN sanitization: zero out any remaining NaN in obs/reward
        obs_nan_mask = torch.isnan(self._obs_buf).any(dim=-1)
        if obs_nan_mask.any():
            self._obs_buf[obs_nan_mask] = 0.0
        reward_nan_mask = torch.isnan(self._reward_buf)
        if reward_nan_mask.any():
            self._reward_buf[reward_nan_mask] = 0.0

        return self._obs_buf, self._reward_buf, self._done_buf, self._info

    def _pre_physics_step(self, actions):
        """Apply actions as joint position targets to the simulation."""
        self._robot.set_joint_position_target(actions)
        self._robot.write_data_to_sim()

        if self.has_dm_envs():
            self._dm_env.pre_physics_step()

    # ==================== Observations ====================

    def _update_observations(self, env_ids=None):
        """Compute observations and write to obs buffer."""
        obs = self._compute_obs(env_ids)
        if env_ids is None:
            self._obs_buf[:] = obs
        elif len(env_ids) > 0:
            self._obs_buf[env_ids] = obs

    def _compute_obs(self, env_ids=None, ret_obs_shapes=False):
        """Compute observations matching the original IGParkourEnv format."""
        if env_ids is None:
            dm_env_ids = None
            root_pos = self._char_root_pos
            root_rot = self._char_root_rot
            root_vel = self._char_root_vel
            root_ang_vel = self._char_root_ang_vel
            dof_pos = self._char_dof_pos
            dof_vel = self._char_dof_vel
            if self._use_heightmap:
                hf = self._ray_hfs
        else:
            dm_env_ids = self._extract_dm_env_ids(env_ids)
            root_pos = self._char_root_pos[env_ids]
            root_rot = self._char_root_rot[env_ids]
            root_vel = self._char_root_vel[env_ids]
            root_ang_vel = self._char_root_ang_vel[env_ids]
            dof_pos = self._char_dof_pos[env_ids]
            dof_vel = self._char_dof_vel[env_ids]
            if self._use_heightmap:
                hf = self._ray_hfs[env_ids]

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        if self._has_key_bodies():
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        # Get target observations from dm_env
        tar_root_pos = []
        tar_root_rot = []
        tar_joint_rot = []
        tar_key_pos = []
        tar_contacts = []

        if self.has_dm_envs() and (dm_env_ids is None or len(dm_env_ids) > 0):
            dm_tar_root_pos, dm_tar_root_rot, dm_tar_joint_rot, dm_tar_key_pos, dm_tar_contacts = \
                self._dm_env.compute_tar_obs(self._tar_obs_steps, dm_env_ids)
            tar_root_pos.append(dm_tar_root_pos)
            tar_root_rot.append(dm_tar_root_rot)
            tar_joint_rot.append(dm_tar_joint_rot)
            tar_key_pos.append(dm_tar_key_pos)
            tar_contacts.append(dm_tar_contacts)

        tar_root_pos = torch.cat(tar_root_pos, dim=0)
        tar_root_rot = torch.cat(tar_root_rot, dim=0)
        tar_joint_rot = torch.cat(tar_joint_rot, dim=0)
        tar_key_pos = torch.cat(tar_key_pos, dim=0)
        tar_contacts = torch.cat(tar_contacts, dim=0)

        obs_dict = mgdm_dm_util.compute_deepmimic_obs(
            root_pos=root_pos, root_rot=root_rot,
            root_vel=root_vel, root_ang_vel=root_ang_vel,
            joint_rot=joint_rot, dof_vel=dof_vel,
            key_pos=key_pos, global_obs=self._global_obs,
            root_height_obs=self._global_root_height_obs,
            enable_tar_obs=self._enable_tar_obs,
            tar_root_pos=tar_root_pos, tar_root_rot=tar_root_rot,
            tar_joint_rot=tar_joint_rot, tar_key_pos=tar_key_pos)

        obs = []
        obs_shapes = OrderedDict()
        for key in obs_dict:
            curr_obs = obs_dict[key]
            if ret_obs_shapes:
                obs_shapes[key] = {"use_normalizer": True, "shape": curr_obs.shape[1:]}
            if key == "tar_obs":
                curr_obs = torch.reshape(curr_obs, [curr_obs.shape[0], curr_obs.shape[1] * curr_obs.shape[2]])
            obs.append(curr_obs)

        if self._use_contact_info:
            if self._enable_tar_obs:
                tar_contacts_flat = tar_contacts.reshape(
                    tar_contacts.shape[0], tar_contacts.shape[1] * tar_contacts.shape[2])
                obs.append(tar_contacts_flat)

            char_contacts = self._get_char_contact_state(env_ids)
            obs.append(char_contacts)

            if ret_obs_shapes:
                if self._enable_tar_obs:
                    obs_shapes["tar_contacts"] = {"use_normalizer": False, "shape": tar_contacts.shape[1:]}
                obs_shapes["char_contacts"] = {"use_normalizer": False, "shape": char_contacts.shape[1:]}

        if self._use_heightmap:
            obs.append(hf)
            if ret_obs_shapes:
                obs_shapes["hf"] = {"use_normalizer": False, "shape": self._ray_hfs.shape[1:]}

        if ret_obs_shapes:
            return obs_shapes

        obs = torch.cat(obs, dim=-1)
        return obs

    # ==================== Reward ====================

    def _update_reward(self):
        """Compute DeepMimic reward."""
        joint_rot = self._kin_char_model.dof_to_rot(self._char_dof_pos)
        if self._has_key_bodies():
            key_pos = self._char_rigid_body_pos[..., self._key_body_ids, :]
            ref_key_pos = self._ref_body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)
            ref_key_pos = key_pos

        comp_r = mgdm_dm_util.compute_deepmimic_reward(
            root_pos=self._char_root_pos, root_rot=self._char_root_rot,
            root_vel=self._char_root_vel, root_ang_vel=self._char_root_ang_vel,
            joint_rot=joint_rot, dof_vel=self._char_dof_vel,
            key_pos=key_pos,
            tar_root_pos=self._ref_root_pos, tar_root_rot=self._ref_root_rot,
            tar_root_vel=self._ref_root_vel, tar_root_ang_vel=self._ref_root_ang_vel,
            tar_joint_rot=self._ref_joint_rot, tar_dof_vel=self._ref_dof_vel,
            tar_key_pos=ref_key_pos,
            joint_rot_err_w=self._joint_err_w, dof_err_w=self._dof_err_w,
            track_root_h=self._track_root_h, track_root=self._track_root)

        pose_r, vel_r, root_pos_r, root_vel_r, key_pos_r = \
            comp_r[:, 0], comp_r[:, 1], comp_r[:, 2], comp_r[:, 3], comp_r[:, 4]

        self._info["rewards"] = {
            "pose_r": pose_r, "vel_r": vel_r,
            "root_pos_r": root_pos_r, "root_vel_r": root_vel_r,
            "key_pos_r": key_pos_r,
        }

        deepmimic_r = (self._pose_w * pose_r + self._vel_w * vel_r
                       + self._root_pos_w * root_pos_r + self._root_vel_w * root_vel_r
                       + self._key_pos_w * key_pos_r)

        if self._use_contact_info:
            char_contact_forces = self._char_contact_forces[:, self._kin_char_model.get_contact_body_ids()]
            comp_penalty = mgdm_dm_util.compute_contact_reward(
                tar_contacts=self._ref_contacts,
                contact_forces=char_contact_forces,
                contact_weights=self._contact_weights)
            contact_penalty = torch.mean(comp_penalty, dim=-1)
            self._info["rewards"]["contact_penalty"] = contact_penalty
            deepmimic_r += contact_penalty

        self._reward_buf[:] = deepmimic_r
        self._info["rewards"]["total_r"] = self._reward_buf.clone()

        if self._report_tracking_error and self.has_dm_envs():
            char_joint_rot = self._kin_char_model.dof_to_rot(self._char_dof_pos)
            char_body_pos, char_body_rot = self._kin_char_model.forward_kinematics(
                self._char_root_pos, self._char_root_rot, char_joint_rot)
            ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(
                self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot)
            tracking_error = mgdm_dm_util.compute_tracking_error(
                root_pos=self._char_root_pos, root_rot=self._char_root_rot,
                body_rot=char_body_rot, body_pos=char_body_pos,
                tar_root_pos=self._ref_root_pos, tar_root_rot=self._ref_root_rot,
                tar_body_rot=ref_body_rot, tar_body_pos=ref_body_pos,
                root_vel=self._char_root_vel, root_ang_vel=self._char_root_ang_vel,
                dof_vel=self._char_dof_vel,
                tar_root_vel=self._ref_root_vel, tar_root_ang_vel=self._ref_root_ang_vel,
                tar_dof_vel=self._ref_dof_vel)
            self._info["tracking_error"] = tracking_error

    # ==================== Termination ====================

    def _update_done(self):
        """Compute termination flags via dm_env."""
        if self.has_dm_envs():
            self._dm_env.update_done(
                termination_height=self._termination_height,
                episode_length=self._episode_length,
                contact_body_ids=self._contact_body_ids,
                pose_termination=self._pose_termination,
                pose_termination_dist=self._pose_termination_dist,
                global_obs=self._global_obs,
                enable_early_termination=self._enable_early_termination,
                track_root=self._track_root,
                root_pos_termination_dist=self._root_pos_termination_dist,
                root_rot_termination_angle=self._root_rot_termination_angle)

    # ==================== Info / Heightmap ====================

    def _update_info(self, env_ids=None):
        self._info["timestep"] = self._timestep_buf.clone().detach()
        compute_time = time.time() - self._start_compute_time
        self._info["ep_num"] = self._ep_num_buf.detach().clone()
        self._info["compute_time"] = compute_time
        self._info["char_contact_forces"] = self._char_contact_forces.detach().clone()

    def _refresh_obs_hfs(self):
        """Refresh heightmap ray observations."""
        if not self._use_heightmap:
            return
        char_root_pos_xyz = self._get_global_xyz_pos(self._char_root_pos[:, 0:3])
        char_heading = torch_util.calc_heading(self._char_root_rot)

        if self.has_dm_envs():
            dm_char_root_pos_xyz = self._get_dm_slice(char_root_pos_xyz)
            dm_char_heading = self._get_dm_slice(char_heading)
            self._dm_env._refresh_obs_hfs(dm_char_root_pos_xyz, dm_char_heading)

    def _get_char_contact_state(self, env_ids, eps=None):
        if eps is None:
            eps = self._contact_detection_eps
        if env_ids is None:
            char_contacts = torch.norm(self._char_contact_forces, dim=-1)
        else:
            char_contacts = torch.norm(self._char_contact_forces[env_ids], dim=-1)
        char_contacts = (char_contacts > eps).to(dtype=torch.float32)
        return char_contacts

    def _get_char_state(self, env_ids, eps=1e-5, concat=False, ref=False):
        if ref:
            root_pos = self._ref_root_pos[env_ids]
            root_rot = torch_util.quat_to_exp_map(self._ref_root_rot[env_ids])
            joint_dof = self._ref_dof_pos[env_ids]
            char_contacts = self._ref_contacts[env_ids]
        else:
            root_pos = self._char_root_pos[env_ids]
            root_rot = torch_util.quat_to_exp_map(self._char_root_rot[env_ids])
            joint_dof = self._char_dof_pos[env_ids]
            char_contacts = torch.norm(self._char_contact_forces[env_ids], dim=-1)
            char_contacts = (char_contacts > eps).to(dtype=torch.float32)

        char_state = torch.cat([root_pos, root_rot, joint_dof], dim=-1)
        ret = [char_state, char_contacts]
        if concat:
            return torch.cat(ret, dim=-1)
        else:
            return tuple(ret)

    # ==================== Helpers ====================

    def _has_key_bodies(self):
        return len(self._key_body_ids) > 0

    def _parse_joint_err_weights(self, joint_err_w):
        num_joints = self._kin_char_model.get_num_joints()
        if joint_err_w is None:
            self._joint_err_w = torch.ones(num_joints - 1, device=self._device, dtype=torch.float32)
        else:
            self._joint_err_w = torch.tensor(joint_err_w, device=self._device, dtype=torch.float32)
        assert self._joint_err_w.shape[-1] == num_joints - 1

        dof_size = self._kin_char_model.get_dof_size()
        self._dof_err_w = torch.zeros(dof_size, device=self._device, dtype=torch.float32)
        for j in range(1, num_joints):
            dof_dim = self._kin_char_model.get_joint_dof_dim(j)
            if dof_dim > 0:
                curr_w = self._joint_err_w[j - 1]
                dof_idx = self._kin_char_model.get_joint_dof_idx(j)
                self._dof_err_w[dof_idx:dof_idx + dof_dim] = curr_w

    def _get_global_xy_pos(self, env_pos, env_ids=None):
        if env_ids is None:
            return env_pos + self._env_offsets[:, 0:2]
        else:
            return env_pos + self._env_offsets[env_ids, 0:2]

    def _get_global_xyz_pos(self, env_pos, env_ids=None):
        if env_ids is None:
            return env_pos + self._env_offsets[:, 0:3]
        else:
            return env_pos + self._env_offsets[env_ids, 0:3]

    def _extract_dm_env_ids(self, env_ids):
        return env_ids[env_ids < self._num_dm_envs]

    def _get_dm_slice(self, data):
        assert data.shape[0] == self._num_envs
        return data[:self._num_dm_envs]

    def has_dm_envs(self):
        return self._num_dm_envs > 0

    def get_dm_env(self):
        return self._dm_env

    def get_extra_log_info(self):
        extra_log_info = {}
        if self.has_dm_envs():
            extra_log_info.update(self._dm_env.get_extra_log_info())
        return extra_log_info

    def post_test_update(self):
        if self.has_dm_envs():
            self._dm_env.post_test_update()

    def close(self):
        self._mjlab_env.close()

    # ==================== Motion Recording ====================

    def set_write_agent_states_flag(self, val):
        self._write_agent_states_flag = val

    def is_writing_agent_states(self):
        return self._write_agent_states_flag

    def is_writing_env_state(self, env_id):
        return self._writing_env_state[env_id]

    def set_writing_env_state(self, env_id, val):
        self._writing_env_state[env_id] = val

    def set_env_success_state(self, env_id, val):
        self._env_success_state[env_id] = val

    def get_env_success_states(self):
        return self._env_success_state

    def build_agent_states_dict(self, name_suffix="", record_obs=False):
        if not record_obs:
            self._dm_agent_motion = [{
                "fps": int(self._control_freq),
                "loop_mode": "CLAMP",
                "frames": [], "contacts": []
            } for _ in range(self._num_envs)]
        else:
            env_ids = torch.tensor([0], dtype=torch.int64, device=self._device)
            obs_shapes = self._compute_obs(env_ids, ret_obs_shapes=True)
            self._dm_agent_motion = [{
                "fps": int(self._control_freq),
                "loop_mode": "CLAMP",
                "frames": [], "contacts": [], "obs": [],
                "obs_shapes": obs_shapes
            } for _ in range(self._num_envs)]
        self._record_obs = record_obs
        self.set_write_agent_states_flag(True)
        self._writing_env_state = [True] * self._num_envs
        self._env_success_state = [False] * self._num_envs
        self._save_motion_name_suffix = name_suffix

    def write_agent_states(self):
        if not self.is_writing_agent_states():
            return

        self.set_write_agent_states_flag(False)
        for env_id in range(self._num_envs):
            if not self.is_writing_env_state(env_id):
                continue
            self.set_write_agent_states_flag(True)

            ref = hasattr(self, "_record_ref") and self._record_ref is True
            char_states, char_contacts = self._get_char_state(env_id, ref=ref)
            self._dm_agent_motion[env_id]["frames"].append(char_states.detach().clone())
            self._dm_agent_motion[env_id]["contacts"].append(char_contacts.detach().clone())

            if self._record_obs:
                self._dm_agent_motion[env_id]["obs"].append(self._obs_buf[env_id].detach().clone())

            if self._done_buf[env_id] == base_env.DoneFlags.FAIL.value:
                self.set_writing_env_state(env_id, False)
                if self.has_dm_envs() and env_id < self._num_dm_envs:
                    motion_length = self._dm_env.get_env_motion_length(env_id).item()
                    curr_motion_time = self._dm_env.get_env_motion_time(env_id).item()
                    motion_name = self._dm_env.get_env_motion_name(env_id)

                    if not self._bypass_record_fail:
                        if curr_motion_time < motion_length - self._timestep * 2.0:
                            continue

                    self.set_env_success_state(env_id, True)
                    output_motion_name = motion_name + self._save_motion_name_suffix
                    self.save_agent_states_to_file(env_id, output_motion_name)
                else:
                    self.save_agent_states_to_file(env_id)

    def save_agent_states_to_file(self, env_id, output_motion_name=None):
        motion_frames = torch.stack(self._dm_agent_motion[env_id]["frames"])
        contact_frames = torch.stack(self._dm_agent_motion[env_id]["contacts"])

        if self._record_obs:
            obs = torch.stack(self._dm_agent_motion[env_id]["obs"])
            self._dm_agent_motion[env_id]["obs"] = obs.cpu().numpy()

        if env_id < self._num_dm_envs:
            motion_frames[:, 0:2] = self._get_global_xy_pos(motion_frames[:, 0:2], env_id)

        if self._use_heightmap and self.has_dm_envs() and env_id < self._num_dm_envs:
            terrain = self._dm_env._terrain
            padding = round(1.0 // terrain.dxdy[0].item()) * terrain.dxdy[0].item()
            sliced_terrain, localized_root_pos = terrain_util.slice_terrain_around_motion(
                motion_frames[..., 0:3], terrain, padding=padding)
            motion_frames = motion_frames.clone()
            motion_frames[..., 0:3] = localized_root_pos
            self._dm_agent_motion[env_id]["terrain"] = sliced_terrain.numpy_copy()

        self._dm_agent_motion[env_id]["frames"] = motion_frames.cpu().numpy()
        self._dm_agent_motion[env_id]["contacts"] = contact_frames.cpu().numpy()

        if output_motion_name is None:
            output_motion_name = "dm_motion_" + str(env_id).zfill(3)

        output_filepath = os.path.join(self._output_motion_dir, output_motion_name + ".pkl")
        with open(output_filepath, 'wb') as file:
            pickle.dump(self._dm_agent_motion[env_id], file)
            Logger.print("wrote motion data to " + output_filepath)

    # ==================== Config Methods ====================

    def set_rand_reset(self, val=None):
        if val is None:
            val = not self._rand_reset
        self._rand_reset = val
        if self.has_dm_envs():
            self._dm_env._rand_reset = val

    def set_demo_mode(self, val=None):
        if self.has_dm_envs():
            self._dm_env.set_demo_mode(val)

    def set_output_motion_dir(self, path):
        self._output_motion_dir = path

    def set_reset_motion_start_time_fraction(self, val):
        if self.has_dm_envs():
            self._dm_env.set_motion_start_time_fraction(val)

    def set_rand_root_pos_offset_scale(self, val):
        if self.has_dm_envs():
            self._dm_env.set_rand_root_pos_offset_scale(val)
