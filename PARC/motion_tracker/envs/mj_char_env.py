"""
mjlab Character Environment for PARC Motion Tracking.

This module implements the PARC motion tracking environment using mjlab's
Manager-Based RL API with MuJoCo Warp backend.

Key Architecture Changes from Isaac Gym (ig_char_env.py):
    1. Monolithic class → Modular Manager-based configuration
    2. gymtorch.wrap_tensor → env.scene.data direct tensor access
    3. Manual physics stepping → mjlab's built-in simulation loop
    4. Custom reset logic → Event terms with MotionLib integration

Tensor Access Translation:
    Isaac Gym (gymtorch):
        root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        self._root_state = gymtorch.wrap_tensor(root_state_tensor)

    mjlab (MuJoCo Warp):
        self._root_state = env.scene.data.root_state_w  # Direct PyTorch tensor
        # Or access via articulation:
        # env.scene["robot"].data.root_pos_w
        # env.scene["robot"].data.root_quat_w

References:
    - Isaac Lab: https://isaac-sim.github.io/IsaacLab/
    - mjlab: https://github.com/mujocolab/mjlab
    - Original PARC: PARC/motion_tracker/envs/ig_char_env.py
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# mjlab imports
from mjlab.envs import ManagerBasedRLEnv
from mjlab.managers import (
    ActionManager,
    EventManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from mjlab.actuators import ActuatorBase
from mjlab.assets import Articulation
from mjlab.scene import InteractiveScene

# PARC imports (PyTorch-based, kept as-is)
import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.torch_util as torch_util
from parc.motion_tracker.envs import base_env

# Local configuration
from parc.motion_tracker.envs.mj_env_config import (
    MJCharEnvCfg,
    MJParkourEnvCfg,
    ControlMode,
    CameraMode,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRLEnvCfg


# =============================================================================
# Custom Actuators
# =============================================================================

class PDExpActuator(ActuatorBase):
    """
    Custom actuator for exponential map PD control.

    This implements the PD control from ig_char_env._calc_pd_exp_torque() for
    3D spherical joints represented as exponential maps.

    In Isaac Gym:
        sim_joint_rot = kin_char_model.dof_to_rot(sim_dof)
        tar_joint_rot = kin_char_model.dof_to_rot(tar_dof)
        diff_dof = kin_char_model.compute_dof_vel(sim_joint_rot, tar_joint_rot, 1.0)
        torque = kp * diff_dof - kd * sim_dof_vel
        torque = clip(torque, -torque_lim, torque_lim)

    In mjlab:
        - Access current state via self._asset.data.joint_pos / joint_vel
        - Compute torque and return it
    """

    def __init__(
        self,
        cfg: Any,
        joint_names: List[str],
        joint_ids: torch.Tensor,
        num_envs: int,
        device: str,
        kin_char_model: kin_char_model.KinCharModel,
    ):
        super().__init__(cfg, joint_names, joint_ids, num_envs, device)
        self._kin_char_model = kin_char_model

        # Extract PD gains from config
        self._kp = self._parse_gains(cfg.stiffness, joint_names)
        self._kd = self._parse_gains(cfg.damping, joint_names)
        self._torque_lim = self._parse_gains(cfg.effort_limit, joint_names)

        # Target buffer (set by action manager)
        self._target_dof = torch.zeros(num_envs, len(joint_ids), device=device)

    def _parse_gains(self, gains: float | dict, joint_names: List[str]) -> torch.Tensor:
        """Parse gains from config (scalar or per-joint dict)."""
        if isinstance(gains, (int, float)):
            return torch.full((len(joint_names),), gains, device=self._device)
        else:
            # Dict mapping joint name patterns to values
            gain_tensor = torch.zeros(len(joint_names), device=self._device)
            for i, name in enumerate(joint_names):
                for pattern, value in gains.items():
                    if pattern in name:
                        gain_tensor[i] = value
                        break
            return gain_tensor

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset actuator state for specified environments."""
        self._target_dof[env_ids] = 0.0

    def set_target(self, target: torch.Tensor) -> None:
        """Set target DOF positions (exponential map representation)."""
        self._target_dof[:] = target

    def compute(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute torque using exponential map PD control.

        Args:
            joint_pos: Current joint positions [num_envs, num_dofs]
            joint_vel: Current joint velocities [num_envs, num_dofs]

        Returns:
            torque: Computed torques [num_envs, num_dofs]
        """
        # Convert current and target DOFs to joint rotations (quaternions)
        sim_joint_rot = self._kin_char_model.dof_to_rot(joint_pos)
        tar_joint_rot = self._kin_char_model.dof_to_rot(self._target_dof)

        # Compute DOF difference using rotation difference
        # This properly handles 3D spherical joints
        diff_dof = self._kin_char_model.compute_dof_vel(sim_joint_rot, tar_joint_rot, 1.0)

        # PD control law
        torque = self._kp * diff_dof - self._kd * joint_vel

        # Apply torque limits
        torque = torch.clamp(torque, -self._torque_lim, self._torque_lim)

        return torque


class PD1DActuator(ActuatorBase):
    """
    Custom actuator for 1D joint PD control.

    This implements simple PD control for hinge joints:
        torque = kp * (tar_dof - sim_dof) - kd * sim_dof_vel
    """

    def __init__(
        self,
        cfg: Any,
        joint_names: List[str],
        joint_ids: torch.Tensor,
        num_envs: int,
        device: str,
    ):
        super().__init__(cfg, joint_names, joint_ids, num_envs, device)

        # Extract PD gains
        self._kp = self._parse_gains(cfg.stiffness, joint_names)
        self._kd = self._parse_gains(cfg.damping, joint_names)
        self._torque_lim = self._parse_gains(cfg.effort_limit, joint_names)

        # Target buffer
        self._target_dof = torch.zeros(num_envs, len(joint_ids), device=device)

    def _parse_gains(self, gains: float | dict, joint_names: List[str]) -> torch.Tensor:
        """Parse gains from config."""
        if isinstance(gains, (int, float)):
            return torch.full((len(joint_names),), gains, device=self._device)
        else:
            gain_tensor = torch.zeros(len(joint_names), device=self._device)
            for i, name in enumerate(joint_names):
                for pattern, value in gains.items():
                    if pattern in name:
                        gain_tensor[i] = value
                        break
            return gain_tensor

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset actuator state."""
        self._target_dof[env_ids] = 0.0

    def set_target(self, target: torch.Tensor) -> None:
        """Set target DOF positions."""
        self._target_dof[:] = target

    def compute(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute torque using simple 1D PD control.

        Args:
            joint_pos: Current joint positions [num_envs, num_dofs]
            joint_vel: Current joint velocities [num_envs, num_dofs]

        Returns:
            torque: Computed torques [num_envs, num_dofs]
        """
        # Simple PD control
        torque = self._kp * (self._target_dof - joint_pos) - self._kd * joint_vel

        # Apply torque limits
        torque = torch.clamp(torque, -self._torque_lim, self._torque_lim)

        return torque


# =============================================================================
# Main Environment Class
# =============================================================================

class MJCharEnv(ManagerBasedRLEnv):
    """
    mjlab Character Environment for PARC Motion Tracking.

    This class extends mjlab's ManagerBasedRLEnv to integrate:
        - PARC's KinCharModel for kinematic computations
        - PARC's MotionLib for motion data management
        - Custom PD actuators for exponential map control
        - DeepMimic-style motion tracking

    Key Differences from Isaac Gym (IGCharEnv):
        1. Tensor access: scene.data instead of gymtorch.wrap_tensor
        2. Physics: MuJoCo Warp instead of PhysX
        3. Structure: Manager-based terms instead of monolithic methods
        4. Reset: Event terms instead of _reset_char()

    Tensor Mapping (Isaac Gym → mjlab):
        self._char_root_pos → self.scene["robot"].data.root_pos_w[:, :3]
        self._char_root_rot → self.scene["robot"].data.root_quat_w
        self._char_root_vel → self.scene["robot"].data.root_lin_vel_w
        self._char_root_ang_vel → self.scene["robot"].data.root_ang_vel_w
        self._char_dof_pos → self.scene["robot"].data.joint_pos
        self._char_dof_vel → self.scene["robot"].data.joint_vel
        self._char_contact_forces → self.scene["robot"].data.net_contact_forces_w
    """

    cfg: MJCharEnvCfg

    def __init__(self, cfg: MJCharEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the character environment.

        Args:
            cfg: Environment configuration
            render_mode: Rendering mode ("human", "rgb_array", or None)
            **kwargs: Additional arguments passed to base class
        """
        # Build kinematic character model before parent init
        # (needed for actuator setup and observation computation)
        self._build_kin_char_model(cfg.char_file)

        # Initialize motion library if configured
        self._motion_lib: Optional[motion_lib.MotionLib] = None
        if cfg.motion_file:
            self._build_motion_lib(cfg.motion_file, cfg)

        # Store configuration
        self._control_mode = cfg.control_mode
        self._camera_mode = cfg.camera_mode
        self._global_obs = cfg.global_obs
        self._root_height_obs = cfg.root_height_obs

        # Key body IDs for observations/rewards
        self._key_body_ids: Optional[torch.Tensor] = None

        # Reference state buffers (populated during reset from MotionLib)
        self._ref_root_pos: Optional[torch.Tensor] = None
        self._ref_root_rot: Optional[torch.Tensor] = None
        self._ref_root_vel: Optional[torch.Tensor] = None
        self._ref_root_ang_vel: Optional[torch.Tensor] = None
        self._ref_joint_rot: Optional[torch.Tensor] = None
        self._ref_dof_pos: Optional[torch.Tensor] = None
        self._ref_dof_vel: Optional[torch.Tensor] = None
        self._ref_body_pos: Optional[torch.Tensor] = None
        self._ref_contacts: Optional[torch.Tensor] = None

        # Motion tracking state
        self._motion_ids: Optional[torch.Tensor] = None
        self._motion_times: Optional[torch.Tensor] = None

        # Call parent constructor (builds scene, managers, etc.)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Build additional data buffers after scene is ready
        self._build_data_buffers()

    def _build_kin_char_model(self, char_file: str) -> None:
        """
        Build the kinematic character model from MJCF file.

        This replicates IGCharEnv._build_kin_char_model() functionality.
        """
        if not char_file:
            raise ValueError("char_file must be specified in configuration")

        _, file_ext = os.path.splitext(char_file)
        if file_ext == ".xml":
            self._kin_char_model = kin_char_model.KinCharModel(self.device)
            self._kin_char_model.load_char_file(char_file)
        else:
            raise ValueError(f"Unsupported character file format: {file_ext}")

    def _build_motion_lib(self, motion_file: str, cfg: MJCharEnvCfg) -> None:
        """
        Build the motion library for reference motion tracking.

        This replicates the MotionLib loading from DeepMimicEnv.
        """
        self._motion_lib = motion_lib.MotionLib.from_file(
            motion_file=motion_file,
            char_model=self._kin_char_model,
            device=self.device,
            contact_info=cfg.use_contact_info,
        )

    def _build_data_buffers(self) -> None:
        """
        Build additional data buffers after scene initialization.

        This creates the reference state buffers and other tensors
        needed for motion tracking.
        """
        num_envs = self.num_envs
        device = self.device

        # Get robot articulation
        robot: Articulation = self.scene["robot"]
        num_bodies = robot.num_bodies
        num_dofs = robot.num_joints

        # Reference state buffers
        self._ref_root_pos = torch.zeros(num_envs, 3, device=device)
        self._ref_root_rot = torch.zeros(num_envs, 4, device=device)
        self._ref_root_rot[:, 3] = 1.0  # Identity quaternion (w=1)
        self._ref_root_vel = torch.zeros(num_envs, 3, device=device)
        self._ref_root_ang_vel = torch.zeros(num_envs, 3, device=device)

        num_joints = self._kin_char_model.get_num_joints()
        self._ref_joint_rot = torch.zeros(num_envs, num_joints - 1, 4, device=device)
        self._ref_joint_rot[:, :, 3] = 1.0

        self._ref_dof_pos = torch.zeros(num_envs, num_dofs, device=device)
        self._ref_dof_vel = torch.zeros(num_envs, num_dofs, device=device)
        self._ref_body_pos = torch.zeros(num_envs, num_bodies, 3, device=device)

        if self.cfg.use_contact_info:
            num_contact_bodies = self._kin_char_model.get_num_contact_bodies()
            self._ref_contacts = torch.zeros(num_envs, num_contact_bodies, device=device)

        # Motion tracking state
        self._motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._motion_times = torch.zeros(num_envs, device=device)

        # Build key body IDs tensor
        if self.cfg.key_bodies:
            key_body_ids = []
            for body_name in self.cfg.key_bodies:
                body_id = self._kin_char_model.get_body_id(body_name)
                key_body_ids.append(body_id)
            self._key_body_ids = torch.tensor(key_body_ids, dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # Property accessors for tensor data (mjlab-style)
    # -------------------------------------------------------------------------

    @property
    def robot(self) -> Articulation:
        """Get the robot articulation."""
        return self.scene["robot"]

    @property
    def char_root_pos(self) -> torch.Tensor:
        """
        Character root position [num_envs, 3].

        Isaac Gym equivalent: self._char_root_pos
        mjlab access: self.scene["robot"].data.root_pos_w[:, :3]
        """
        return self.robot.data.root_pos_w

    @property
    def char_root_rot(self) -> torch.Tensor:
        """
        Character root rotation (quaternion wxyz) [num_envs, 4].

        Isaac Gym equivalent: self._char_root_rot
        mjlab access: self.scene["robot"].data.root_quat_w
        """
        return self.robot.data.root_quat_w

    @property
    def char_root_vel(self) -> torch.Tensor:
        """
        Character root linear velocity [num_envs, 3].

        Isaac Gym equivalent: self._char_root_vel
        mjlab access: self.scene["robot"].data.root_lin_vel_w
        """
        return self.robot.data.root_lin_vel_w

    @property
    def char_root_ang_vel(self) -> torch.Tensor:
        """
        Character root angular velocity [num_envs, 3].

        Isaac Gym equivalent: self._char_root_ang_vel
        mjlab access: self.scene["robot"].data.root_ang_vel_w
        """
        return self.robot.data.root_ang_vel_w

    @property
    def char_dof_pos(self) -> torch.Tensor:
        """
        Character joint positions [num_envs, num_dofs].

        Isaac Gym equivalent: self._char_dof_pos
        mjlab access: self.scene["robot"].data.joint_pos
        """
        return self.robot.data.joint_pos

    @property
    def char_dof_vel(self) -> torch.Tensor:
        """
        Character joint velocities [num_envs, num_dofs].

        Isaac Gym equivalent: self._char_dof_vel
        mjlab access: self.scene["robot"].data.joint_vel
        """
        return self.robot.data.joint_vel

    @property
    def char_body_pos(self) -> torch.Tensor:
        """
        Character body positions [num_envs, num_bodies, 3].

        Isaac Gym equivalent: self._char_rigid_body_pos
        mjlab access: self.scene["robot"].data.body_pos_w
        """
        return self.robot.data.body_pos_w

    @property
    def char_body_rot(self) -> torch.Tensor:
        """
        Character body rotations [num_envs, num_bodies, 4].

        Isaac Gym equivalent: self._char_rigid_body_rot
        mjlab access: self.scene["robot"].data.body_quat_w
        """
        return self.robot.data.body_quat_w

    @property
    def char_contact_forces(self) -> torch.Tensor:
        """
        Contact forces on character bodies [num_envs, num_bodies, 3].

        Isaac Gym equivalent: self._char_contact_forces
        mjlab access: self.scene["robot"].data.net_contact_forces_w
        """
        return self.robot.data.net_contact_forces_w

    # -------------------------------------------------------------------------
    # Reference state accessors
    # -------------------------------------------------------------------------

    @property
    def ref_root_pos(self) -> torch.Tensor:
        """Reference root position from motion library."""
        return self._ref_root_pos

    @property
    def ref_root_rot(self) -> torch.Tensor:
        """Reference root rotation from motion library."""
        return self._ref_root_rot

    @property
    def ref_joint_rot(self) -> torch.Tensor:
        """Reference joint rotations from motion library."""
        return self._ref_joint_rot

    @property
    def ref_dof_pos(self) -> torch.Tensor:
        """Reference DOF positions from motion library."""
        return self._ref_dof_pos

    @property
    def ref_body_pos(self) -> torch.Tensor:
        """Reference body positions from motion library."""
        return self._ref_body_pos

    # -------------------------------------------------------------------------
    # Motion Library Integration
    # -------------------------------------------------------------------------

    def sample_motion_state(self, env_ids: torch.Tensor) -> None:
        """
        Sample motion state from the motion library for given environments.

        This replicates the motion sampling logic from DeepMimicEnv and
        populates the reference state buffers.

        Args:
            env_ids: Environment indices to sample for
        """
        if self._motion_lib is None:
            return

        num_sample = len(env_ids)

        # Sample motion IDs
        motion_ids = self._motion_lib.sample_motions(num_sample)
        self._motion_ids[env_ids] = motion_ids

        # Sample start times
        motion_times = self._motion_lib.sample_time(motion_ids)
        self._motion_times[env_ids] = motion_times

        # Get motion frame data
        frame_data = self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        if self.cfg.use_contact_info:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = frame_data
            self._ref_contacts[env_ids] = contacts
        else:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = frame_data

        # Store reference state
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel

        # Convert joint rotations to DOF
        self._ref_dof_pos[env_ids] = self._kin_char_model.rot_to_dof(joint_rot)

        # Compute reference body positions via forward kinematics
        body_pos, _ = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot
        )
        self._ref_body_pos[env_ids] = body_pos

    def update_motion_time(self, dt: float) -> None:
        """
        Update motion time for all environments.

        Args:
            dt: Time step (typically self.step_dt)
        """
        if self._motion_lib is None:
            return

        self._motion_times += dt

        # Update reference state from motion library at current time
        frame_data = self._motion_lib.calc_motion_frame(
            self._motion_ids, self._motion_times
        )

        if self.cfg.use_contact_info:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = frame_data
            self._ref_contacts[:] = contacts
        else:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = frame_data

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel
        self._ref_dof_pos[:] = self._kin_char_model.rot_to_dof(joint_rot)

        body_pos, _ = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot
        )
        self._ref_body_pos[:] = body_pos

    def get_target_obs(
        self,
        tar_obs_steps: List[int],
        env_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get target observations at future timesteps.

        This replicates compute_tar_obs from ig_parkour_env.

        Args:
            tar_obs_steps: List of future timestep offsets
            env_ids: Environment indices (None for all)

        Returns:
            tar_root_pos: Target root positions [num_envs, num_steps, 3]
            tar_root_rot: Target root rotations [num_envs, num_steps, 4]
            tar_joint_rot: Target joint rotations [num_envs, num_steps, num_joints-1, 4]
            tar_key_pos: Target key body positions [num_envs, num_steps, num_key_bodies, 3]
        """
        if self._motion_lib is None:
            raise RuntimeError("Motion library not initialized")

        if env_ids is None:
            motion_ids = self._motion_ids
            motion_times = self._motion_times
        else:
            motion_ids = self._motion_ids[env_ids]
            motion_times = self._motion_times[env_ids]

        num_envs = motion_ids.shape[0]
        num_steps = len(tar_obs_steps)

        tar_root_pos_list = []
        tar_root_rot_list = []
        tar_joint_rot_list = []
        tar_key_pos_list = []

        for step in tar_obs_steps:
            future_time = motion_times + step * self.step_dt

            frame_data = self._motion_lib.calc_motion_frame(motion_ids, future_time)
            root_pos, root_rot, _, _, joint_rot, _ = frame_data[:6]

            tar_root_pos_list.append(root_pos)
            tar_root_rot_list.append(root_rot)
            tar_joint_rot_list.append(joint_rot)

            # Compute key body positions
            if self._key_body_ids is not None and len(self._key_body_ids) > 0:
                body_pos, _ = self._kin_char_model.forward_kinematics(
                    root_pos, root_rot, joint_rot
                )
                key_pos = body_pos[:, self._key_body_ids, :]
            else:
                key_pos = torch.zeros(num_envs, 0, 3, device=self.device)

            tar_key_pos_list.append(key_pos)

        tar_root_pos = torch.stack(tar_root_pos_list, dim=1)
        tar_root_rot = torch.stack(tar_root_rot_list, dim=1)
        tar_joint_rot = torch.stack(tar_joint_rot_list, dim=1)
        tar_key_pos = torch.stack(tar_key_pos_list, dim=1)

        return tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos

    # -------------------------------------------------------------------------
    # Environment Step Override
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Pre-physics step processing.

        This is called before each physics step to process actions.
        For custom PD control modes, we update the actuator targets here.
        """
        # Call parent implementation
        super()._pre_physics_step(actions)

        # Update motion time for reference tracking
        # Note: This updates the reference state that rewards/observations use
        self.update_motion_time(self.step_dt)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Reset specific environments.

        This overrides the default reset to integrate with MotionLib.
        The state is sampled from the motion library and written to simulation.

        Args:
            env_ids: Environment indices to reset
        """
        if len(env_ids) == 0:
            return

        # Sample motion state from library
        self.sample_motion_state(env_ids)

        # Set robot state from reference
        robot = self.robot

        # Set root state
        robot.write_root_pose_to_sim(
            root_pos=self._ref_root_pos[env_ids],
            root_quat=self._ref_root_rot[env_ids],
            env_ids=env_ids,
        )
        robot.write_root_velocity_to_sim(
            root_lin_vel=self._ref_root_vel[env_ids],
            root_ang_vel=self._ref_root_ang_vel[env_ids],
            env_ids=env_ids,
        )

        # Set joint state
        robot.write_joint_state_to_sim(
            joint_pos=self._ref_dof_pos[env_ids],
            joint_vel=self._ref_dof_vel[env_ids],
            env_ids=env_ids,
        )

        # Call parent reset (handles other managers)
        super()._reset_idx(env_ids)


# =============================================================================
# Observation Term Functions
# =============================================================================
# These are registered with the observation manager and called during
# observation computation. They access data from the environment.

def root_height(env: MJCharEnv) -> torch.Tensor:
    """
    Root height observation.

    Returns:
        Root height [num_envs, 1]
    """
    return env.char_root_pos[:, 2:3]


def root_rotation_tan_norm(env: MJCharEnv, global_obs: bool = False) -> torch.Tensor:
    """
    Root rotation in tan-norm (6D) representation.

    Args:
        env: Environment instance
        global_obs: If True, use global rotation; else use heading-relative

    Returns:
        Root rotation [num_envs, 6]
    """
    root_rot = env.char_root_rot

    if global_obs:
        rot_obs = torch_util.quat_to_tan_norm(root_rot)
    else:
        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        local_root_rot = torch_util.quat_mul(heading_inv_rot, root_rot)
        rot_obs = torch_util.quat_to_tan_norm(local_root_rot)

    return rot_obs


def root_lin_vel(env: MJCharEnv, global_obs: bool = False) -> torch.Tensor:
    """
    Root linear velocity observation.

    Args:
        env: Environment instance
        global_obs: If True, use global velocity; else use heading-relative

    Returns:
        Root linear velocity [num_envs, 3]
    """
    root_vel = env.char_root_vel

    if not global_obs:
        heading_inv_rot = torch_util.calc_heading_quat_inv(env.char_root_rot)
        root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)

    return root_vel


def root_ang_vel(env: MJCharEnv, global_obs: bool = False) -> torch.Tensor:
    """
    Root angular velocity observation.

    Args:
        env: Environment instance
        global_obs: If True, use global velocity; else use heading-relative

    Returns:
        Root angular velocity [num_envs, 3]
    """
    root_ang_vel = env.char_root_ang_vel

    if not global_obs:
        heading_inv_rot = torch_util.calc_heading_quat_inv(env.char_root_rot)
        root_ang_vel = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)

    return root_ang_vel


def joint_rotation_tan_norm(env: MJCharEnv) -> torch.Tensor:
    """
    Joint rotations in tan-norm (6D) representation.

    Returns:
        Joint rotations flattened [num_envs, num_joints * 6]
    """
    dof_pos = env.char_dof_pos
    joint_rot = env._kin_char_model.dof_to_rot(dof_pos)

    # Flatten: [num_envs, num_joints, 4] -> [num_envs * num_joints, 4]
    flat_joint_rot = joint_rot.view(-1, 4)

    # Convert to tan-norm
    flat_rot_obs = torch_util.quat_to_tan_norm(flat_joint_rot)

    # Reshape back and flatten
    num_joints = joint_rot.shape[1]
    rot_obs = flat_rot_obs.view(env.num_envs, num_joints * 6)

    return rot_obs


def joint_vel(env: MJCharEnv) -> torch.Tensor:
    """
    Joint velocity observation.

    Returns:
        Joint velocities [num_envs, num_dofs]
    """
    return env.char_dof_vel


def key_body_positions(
    env: MJCharEnv,
    key_bodies: List[str],
    global_obs: bool = False,
) -> torch.Tensor:
    """
    Key body positions relative to root.

    Args:
        env: Environment instance
        key_bodies: List of body names
        global_obs: If True, use global positions; else use heading-relative

    Returns:
        Key body positions flattened [num_envs, num_key_bodies * 3]
    """
    if not key_bodies:
        return torch.zeros(env.num_envs, 0, device=env.device)

    # Get key body IDs
    key_body_ids = []
    for name in key_bodies:
        body_id = env._kin_char_model.get_body_id(name)
        key_body_ids.append(body_id)

    # Compute body positions via forward kinematics
    dof_pos = env.char_dof_pos
    joint_rot = env._kin_char_model.dof_to_rot(dof_pos)
    body_pos, _ = env._kin_char_model.forward_kinematics(
        env.char_root_pos, env.char_root_rot, joint_rot
    )

    # Extract key body positions
    key_pos = body_pos[:, key_body_ids, :]

    # Make relative to root
    root_pos = env.char_root_pos.unsqueeze(1)
    key_pos = key_pos - root_pos

    # Transform to heading frame if not global
    if not global_obs:
        heading_inv_rot = torch_util.calc_heading_quat_inv(env.char_root_rot)
        # Expand for broadcasting
        heading_inv_rot = heading_inv_rot.unsqueeze(1).expand(-1, len(key_body_ids), -1)
        flat_rot = heading_inv_rot.reshape(-1, 4)
        flat_pos = key_pos.reshape(-1, 3)
        flat_local_pos = torch_util.quat_rotate(flat_rot, flat_pos)
        key_pos = flat_local_pos.view(env.num_envs, len(key_body_ids), 3)

    # Flatten
    return key_pos.view(env.num_envs, -1)


def character_contacts(env: MJCharEnv, contact_threshold: float = 1e-5) -> torch.Tensor:
    """
    Character contact state observation.

    Returns:
        Binary contact state for each body [num_envs, num_bodies]
    """
    contact_forces = env.char_contact_forces
    contact_mag = torch.norm(contact_forces, dim=-1)
    contacts = (contact_mag > contact_threshold).float()
    return contacts


# =============================================================================
# Reward Term Functions
# =============================================================================

def pose_reward(env: MJCharEnv, scale: float = 2.0) -> torch.Tensor:
    """
    Pose tracking reward based on joint rotation error.

    This computes the quaternion distance between current and reference
    joint rotations.

    Returns:
        Pose reward [num_envs]
    """
    # Get current joint rotations
    dof_pos = env.char_dof_pos
    curr_joint_rot = env._kin_char_model.dof_to_rot(dof_pos)

    # Get reference joint rotations
    ref_joint_rot = env.ref_joint_rot

    # Compute rotation difference
    diff_rot = torch_util.quat_diff(curr_joint_rot, ref_joint_rot)
    diff_angle = torch_util.quat_to_exp_map(diff_rot)
    diff_angle_sq = torch.sum(diff_angle ** 2, dim=-1)

    # Average over joints
    diff_angle_sq_mean = torch.mean(diff_angle_sq, dim=-1)

    # Exponential reward
    reward = torch.exp(-scale * diff_angle_sq_mean)
    return reward


def vel_reward(env: MJCharEnv, scale: float = 0.1) -> torch.Tensor:
    """
    Velocity tracking reward based on joint velocity error.

    Returns:
        Velocity reward [num_envs]
    """
    curr_vel = env.char_dof_vel
    ref_vel = env._ref_dof_vel

    diff_vel = curr_vel - ref_vel
    diff_vel_sq = torch.sum(diff_vel ** 2, dim=-1)

    reward = torch.exp(-scale * diff_vel_sq)
    return reward


def root_pos_reward(
    env: MJCharEnv,
    scale: float = 2.0,
    track_root_h: bool = True,
) -> torch.Tensor:
    """
    Root position tracking reward.

    Args:
        scale: Reward scale
        track_root_h: If True, also track root height

    Returns:
        Root position reward [num_envs]
    """
    curr_pos = env.char_root_pos
    ref_pos = env.ref_root_pos

    if track_root_h:
        diff_pos = curr_pos - ref_pos
    else:
        # Only track XY
        diff_pos = curr_pos[:, :2] - ref_pos[:, :2]

    diff_pos_sq = torch.sum(diff_pos ** 2, dim=-1)

    reward = torch.exp(-scale * diff_pos_sq)
    return reward


def root_vel_reward(env: MJCharEnv, scale: float = 2.0) -> torch.Tensor:
    """
    Root velocity tracking reward.

    Returns:
        Root velocity reward [num_envs]
    """
    curr_vel = env.char_root_vel
    ref_vel = env._ref_root_vel

    diff_vel = curr_vel - ref_vel
    diff_vel_sq = torch.sum(diff_vel ** 2, dim=-1)

    reward = torch.exp(-scale * diff_vel_sq)
    return reward


def key_pos_reward(
    env: MJCharEnv,
    scale: float = 10.0,
    key_bodies: List[str] = None,
) -> torch.Tensor:
    """
    Key body position tracking reward.

    Returns:
        Key position reward [num_envs]
    """
    if not key_bodies or env._key_body_ids is None:
        return torch.ones(env.num_envs, device=env.device)

    # Current key body positions
    dof_pos = env.char_dof_pos
    joint_rot = env._kin_char_model.dof_to_rot(dof_pos)
    body_pos, _ = env._kin_char_model.forward_kinematics(
        env.char_root_pos, env.char_root_rot, joint_rot
    )
    curr_key_pos = body_pos[:, env._key_body_ids, :]

    # Reference key body positions
    ref_key_pos = env.ref_body_pos[:, env._key_body_ids, :]

    # Compute error
    diff_pos = curr_key_pos - ref_key_pos
    diff_pos_sq = torch.sum(diff_pos ** 2, dim=-1)
    diff_pos_sq_mean = torch.mean(diff_pos_sq, dim=-1)

    reward = torch.exp(-scale * diff_pos_sq_mean)
    return reward


def contact_penalty(
    env: MJCharEnv,
    contact_weights: List[float] = None,
) -> torch.Tensor:
    """
    Contact matching penalty.

    Returns:
        Contact penalty [num_envs] (negative values)
    """
    if env._ref_contacts is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Get contact state
    contact_ids = env._kin_char_model.get_contact_body_ids()
    contact_forces = env.char_contact_forces[:, contact_ids, :]
    contact_mag = torch.norm(contact_forces, dim=-1)
    curr_contacts = (contact_mag > 1e-5).float()

    ref_contacts = env._ref_contacts

    # Compute mismatch penalty
    if contact_weights:
        weights = torch.tensor(contact_weights, device=env.device)
        mismatch = torch.abs(curr_contacts - ref_contacts) * weights
    else:
        mismatch = torch.abs(curr_contacts - ref_contacts)

    penalty = -torch.mean(mismatch, dim=-1)
    return penalty


# =============================================================================
# Termination Term Functions
# =============================================================================

def time_out(env: MJCharEnv) -> torch.Tensor:
    """
    Time-out termination.

    Returns:
        Boolean tensor [num_envs] indicating time-out
    """
    return env.episode_length_buf >= env.max_episode_length


def root_height_termination(
    env: MJCharEnv,
    termination_height: float = 0.3,
) -> torch.Tensor:
    """
    Root height termination (character fell).

    Returns:
        Boolean tensor [num_envs] indicating termination
    """
    root_h = env.char_root_pos[:, 2]
    return root_h < termination_height


def pose_deviation_termination(
    env: MJCharEnv,
    pose_termination_dist: float = 1.0,
) -> torch.Tensor:
    """
    Pose deviation termination (tracking error too large).

    Returns:
        Boolean tensor [num_envs] indicating termination
    """
    # Compute pose error
    dof_pos = env.char_dof_pos
    curr_joint_rot = env._kin_char_model.dof_to_rot(dof_pos)
    ref_joint_rot = env.ref_joint_rot

    diff_rot = torch_util.quat_diff(curr_joint_rot, ref_joint_rot)
    diff_angle = torch_util.quat_to_exp_map(diff_rot)
    diff_angle_norm = torch.norm(diff_angle, dim=-1)
    max_diff = torch.max(diff_angle_norm, dim=-1)[0]

    return max_diff > pose_termination_dist


# =============================================================================
# Event Term Functions (Reset)
# =============================================================================

def reset_from_motion_lib(
    env: MJCharEnv,
    env_ids: torch.Tensor,
    motion_file: str,
    rand_reset: bool = True,
    truncate_time: float = None,
) -> None:
    """
    Reset environments from motion library.

    This samples a motion and start time, then sets the initial state.

    Args:
        env: Environment instance
        env_ids: Environment indices to reset
        motion_file: Path to motion file (unused, motion_lib already loaded)
        rand_reset: If True, sample random start time
        truncate_time: Time to truncate from motion end
    """
    if env._motion_lib is None:
        return

    env.sample_motion_state(env_ids)


def reset_root_state(
    env: MJCharEnv,
    env_ids: torch.Tensor,
    asset_cfg: Any,
) -> None:
    """
    Reset root state from reference.

    Args:
        env: Environment instance
        env_ids: Environment indices to reset
        asset_cfg: Asset configuration (unused, using default robot)
    """
    robot = env.robot

    robot.write_root_pose_to_sim(
        root_pos=env._ref_root_pos[env_ids],
        root_quat=env._ref_root_rot[env_ids],
        env_ids=env_ids,
    )
    robot.write_root_velocity_to_sim(
        root_lin_vel=env._ref_root_vel[env_ids],
        root_ang_vel=env._ref_root_ang_vel[env_ids],
        env_ids=env_ids,
    )


def reset_joint_state(
    env: MJCharEnv,
    env_ids: torch.Tensor,
    asset_cfg: Any,
) -> None:
    """
    Reset joint state from reference.

    Args:
        env: Environment instance
        env_ids: Environment indices to reset
        asset_cfg: Asset configuration (unused, using default robot)
    """
    robot = env.robot

    robot.write_joint_state_to_sim(
        joint_pos=env._ref_dof_pos[env_ids],
        joint_vel=env._ref_dof_vel[env_ids],
        env_ids=env_ids,
    )


# =============================================================================
# Module Registration
# =============================================================================
# Register observation, reward, termination, and event functions
# These are referenced by string in the configuration

# Create a module-level registry
mj_char_obs = type("mj_char_obs", (), {
    "root_height": root_height,
    "root_rotation_tan_norm": root_rotation_tan_norm,
    "root_lin_vel": root_lin_vel,
    "root_ang_vel": root_ang_vel,
    "joint_rotation_tan_norm": joint_rotation_tan_norm,
    "joint_vel": joint_vel,
    "key_body_positions": key_body_positions,
    "character_contacts": character_contacts,
})()

mj_char_rewards = type("mj_char_rewards", (), {
    "pose_reward": pose_reward,
    "vel_reward": vel_reward,
    "root_pos_reward": root_pos_reward,
    "root_vel_reward": root_vel_reward,
    "key_pos_reward": key_pos_reward,
    "contact_penalty": contact_penalty,
})()

mj_char_terminations = type("mj_char_terminations", (), {
    "time_out": time_out,
    "root_height_termination": root_height_termination,
    "pose_deviation_termination": pose_deviation_termination,
})()

mj_char_events = type("mj_char_events", (), {
    "reset_from_motion_lib": reset_from_motion_lib,
    "reset_root_state": reset_root_state,
    "reset_joint_state": reset_joint_state,
})()
