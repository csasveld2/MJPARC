"""
mjlab Environment Configuration for PARC Motion Tracking.

This module defines the configuration classes for the PARC motion tracking environment
using mjlab's Manager-Based RL API (Isaac Lab style API with MuJoCo Warp backend).

Translation from Isaac Gym:
    - gymtorch.wrap_tensor → env.scene.data buffers
    - gymapi scene creation → mjlab InteractiveSceneCfg
    - Manual tensor management → Manager-based term system

References:
    - Original: PARC/motion_tracker/envs/ig_char_env.py
    - Target API: mjlab ManagerBasedRLEnvCfg (Isaac Lab compatible)
"""

from __future__ import annotations

import enum
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import torch

# mjlab imports (Isaac Lab compatible API)
from mjlab.envs import ManagerBasedRLEnvCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.scene import InteractiveSceneCfg
from mjlab.sim import SimulationCfg
from mjlab.assets import ArticulationCfg
from mjlab.actuators import ActuatorBaseCfg, ImplicitActuatorCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils import configclass

# PARC imports (keeping existing PyTorch-based modules)
import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.torch_util as torch_util


# =============================================================================
# Enums (ported from ig_char_env.py)
# =============================================================================

class CameraMode(enum.Enum):
    """Camera tracking modes for visualization."""
    still = 0
    track = 1


class ControlMode(enum.Enum):
    """
    Control modes for the character.

    - pd: Position-based PD control (uses internal MuJoCo PD actuators)
    - vel: Velocity control
    - torque: Direct torque control
    - pd_exp: Custom exponential map PD control (for 3D spherical joints)
    - pd_1d: Custom 1D PD control (for hinge joints)
    """
    pd = 0
    vel = 1
    torque = 2
    pd_exp = 3
    pd_1d = 4


# =============================================================================
# Custom Actuator Configuration for PD Exponential Control
# =============================================================================

@configclass
class PDExpActuatorCfg(ActuatorBaseCfg):
    """
    Custom actuator configuration for exponential map PD control.

    This implements the custom PD control from ig_char_env._calc_pd_exp_torque()
    which handles 3D spherical joints represented as exponential maps.

    The torque computation:
        diff_dof = kin_char_model.compute_dof_vel(sim_joint_rot, tar_joint_rot, 1.0)
        torque = kp * diff_dof - kd * sim_dof_vel
        torque = clip(torque, -torque_lim, torque_lim)
    """
    class_type: type = MISSING  # Will be set to PDExpActuator in mj_char_env.py

    # PD gains (will be loaded from MJCF asset)
    stiffness: float | dict[str, float] = 100.0
    damping: float | dict[str, float] = 10.0

    # Torque limits (will be loaded from MJCF actuator properties)
    effort_limit: float | dict[str, float] = 100.0

    # Reference to kinematic character model (for exp_map computations)
    kin_char_model_path: str = ""


@configclass
class PD1DActuatorCfg(ActuatorBaseCfg):
    """
    Custom actuator configuration for 1D joint PD control.

    This implements the custom PD control from ig_char_env._calc_pd_1d_torque()
    for hinge joints:
        torque = kp * (tar_dof - sim_dof) - kd * sim_dof_vel
        torque = clip(torque, -torque_lim, torque_lim)
    """
    class_type: type = MISSING  # Will be set to PD1DActuator in mj_char_env.py

    stiffness: float | dict[str, float] = 100.0
    damping: float | dict[str, float] = 10.0
    effort_limit: float | dict[str, float] = 100.0


# =============================================================================
# Scene Configuration
# =============================================================================

@configclass
class MJCharSceneCfg(InteractiveSceneCfg):
    """
    Scene configuration for the character motion tracking environment.

    This replaces the manual scene building in IGEnv._build_envs() and
    IGCharEnv._build_character().
    """

    # Number of parallel environments
    num_envs: int = 4096
    env_spacing: float = 5.0

    # Ground plane configuration (replaces _build_ground_plane)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material={
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "restitution": 0.0,
        },
    )

    # Character articulation (replaces _load_char_asset and _build_character)
    # Note: The actual MJCF path is set at runtime via char_file config
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Character",
        spawn=None,  # Set at runtime from char_file
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),  # Default spawn height
        ),
    )

    # Reference character for visualization (optional, for debugging)
    ref_robot: ArticulationCfg | None = None


# =============================================================================
# Observation Configuration
# =============================================================================

@configclass
class CharacterObsCfg(ObservationGroupCfg):
    """
    Observation configuration for the character.

    This ports the observation computation from ig_char_env.compute_char_obs().

    Observations include:
        - root_h: Root height (if root_height_obs=True)
        - root_rot_obs: Root rotation (tan-norm representation)
        - root_vel_obs: Root linear velocity
        - root_ang_vel_obs: Root angular velocity
        - joint_rot_obs: Joint rotations (tan-norm representation)
        - dof_vel: Joint velocities
        - key_pos_flat: Key body positions relative to root (optional)
    """

    # Whether to enable observation corruption (noise)
    enable_corruption: bool = False
    concatenate_terms: bool = True

    # Root height observation
    root_height: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.root_height",
        params={},
    )

    # Root rotation (6D tan-norm representation)
    root_rot: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.root_rotation_tan_norm",
        params={"global_obs": False},
    )

    # Root linear velocity
    root_lin_vel: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.root_lin_vel",
        params={"global_obs": False},
    )

    # Root angular velocity
    root_ang_vel: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.root_ang_vel",
        params={"global_obs": False},
    )

    # Joint rotations (6D tan-norm for each joint)
    joint_rot: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.joint_rotation_tan_norm",
        params={},
    )

    # Joint velocities
    joint_vel: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.joint_vel",
        params={},
    )

    # Key body positions (hands, feet, etc.)
    key_body_pos: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.key_body_positions",
        params={"key_bodies": [], "global_obs": False},
    )


@configclass
class TargetObsCfg(ObservationGroupCfg):
    """
    Observation configuration for target/reference motion.

    This ports the target observation computation from ig_parkour_env for
    DeepMimic-style motion tracking.
    """

    enable_corruption: bool = False
    concatenate_terms: bool = True

    # Target observations at future timesteps
    tar_root_pos: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.target_root_pos",
        params={"tar_obs_steps": [1]},
    )

    tar_root_rot: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.target_root_rot",
        params={"tar_obs_steps": [1]},
    )

    tar_joint_rot: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.target_joint_rot",
        params={"tar_obs_steps": [1]},
    )

    tar_key_body_pos: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.target_key_body_pos",
        params={"tar_obs_steps": [1], "key_bodies": []},
    )


@configclass
class ContactObsCfg(ObservationGroupCfg):
    """
    Contact observation configuration.

    Includes reference contact labels and current character contacts.
    """

    enable_corruption: bool = False
    concatenate_terms: bool = True

    # Target contact labels from motion library
    tar_contacts: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.target_contacts",
        params={"tar_obs_steps": [1]},
    )

    # Current character contact forces
    char_contacts: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.character_contacts",
        params={"contact_threshold": 1e-5},
    )


@configclass
class HeightfieldObsCfg(ObservationGroupCfg):
    """
    Local heightfield observation for terrain awareness.
    """

    enable_corruption: bool = False
    concatenate_terms: bool = True

    local_heightfield: ObservationTermCfg = ObservationTermCfg(
        func="mj_char_obs.local_heightfield",
        params={
            "ray_points_behind": 3,
            "ray_points_ahead": 5,
            "ray_num_left": 3,
            "ray_num_right": 3,
            "ray_dx": 0.3,
            "ray_angle": 15.0,
        },
    )


@configclass
class ObservationsCfg:
    """
    Full observation configuration combining all groups.

    Corresponds to _compute_obs in ig_char_env.py and ig_parkour_env.py
    """

    # Policy observations (what the agent sees)
    policy: CharacterObsCfg = CharacterObsCfg()

    # Target motion observations (for imitation learning)
    target: TargetObsCfg = TargetObsCfg()

    # Contact observations (optional)
    contacts: ContactObsCfg | None = None

    # Heightfield observations (optional)
    heightfield: HeightfieldObsCfg | None = None


# =============================================================================
# Reward Configuration
# =============================================================================

@configclass
class RewardsCfg:
    """
    Reward configuration for DeepMimic-style motion tracking.

    This ports the reward computation from:
        - ig_char_env._update_reward() → compute_reward()
        - ig_parkour_env._update_reward() → compute_deepmimic_reward()

    Reward components (from ig_parkour_env):
        - pose_r: Joint pose matching reward
        - vel_r: Joint velocity matching reward
        - root_pos_r: Root position matching reward
        - root_vel_r: Root velocity matching reward
        - key_pos_r: Key body position matching reward
        - contact_penalty: Contact matching penalty (optional)
    """

    # Pose tracking reward (joint rotation error)
    pose: RewardTermCfg = RewardTermCfg(
        func="mj_char_rewards.pose_reward",
        weight=0.5,
        params={"scale": 2.0},
    )

    # Velocity tracking reward (joint velocity error)
    vel: RewardTermCfg = RewardTermCfg(
        func="mj_char_rewards.vel_reward",
        weight=0.1,
        params={"scale": 0.1},
    )

    # Root position tracking reward
    root_pos: RewardTermCfg = RewardTermCfg(
        func="mj_char_rewards.root_pos_reward",
        weight=0.15,
        params={"scale": 2.0, "track_root_h": True},
    )

    # Root velocity tracking reward
    root_vel: RewardTermCfg = RewardTermCfg(
        func="mj_char_rewards.root_vel_reward",
        weight=0.1,
        params={"scale": 2.0},
    )

    # Key body position tracking reward
    key_pos: RewardTermCfg = RewardTermCfg(
        func="mj_char_rewards.key_pos_reward",
        weight=0.15,
        params={"scale": 10.0, "key_bodies": []},
    )

    # Contact matching penalty (optional)
    contact_penalty: RewardTermCfg | None = None


# =============================================================================
# Termination Configuration
# =============================================================================

@configclass
class TerminationsCfg:
    """
    Termination configuration for episode ending conditions.

    Ported from ig_char_env._update_done() and ig_parkour_env._update_done()
    """

    # Time limit termination
    time_out: TerminationTermCfg = TerminationTermCfg(
        func="mj_char_terminations.time_out",
        time_out=True,
        params={},
    )

    # Height-based termination (character fell down)
    root_height: TerminationTermCfg = TerminationTermCfg(
        func="mj_char_terminations.root_height_termination",
        time_out=False,
        params={"termination_height": 0.3},
    )

    # Pose deviation termination (tracking error too large)
    pose_deviation: TerminationTermCfg | None = None

    # Root position deviation termination
    root_pos_deviation: TerminationTermCfg | None = None

    # Root rotation deviation termination
    root_rot_deviation: TerminationTermCfg | None = None


# =============================================================================
# Event Configuration (Reset / Randomization)
# =============================================================================

@configclass
class EventCfg:
    """
    Event configuration for environment resets and randomization.

    This implements the reset logic from:
        - ig_char_env._reset_char()
        - ig_parkour_env.reset() with motion library sampling

    Key events:
        - reset_motion: Sample motion from library and set initial state
        - reset_root: Set root position/rotation from motion frame
        - reset_joints: Set joint positions/velocities from motion frame
    """

    # Motion-based reset (samples from MotionLib)
    reset_motion: EventTermCfg = EventTermCfg(
        func="mj_char_events.reset_from_motion_lib",
        mode="reset",
        params={
            "motion_file": "",  # Set at runtime
            "rand_reset": True,
            "truncate_time": None,
        },
    )

    # Root state reset
    reset_root: EventTermCfg = EventTermCfg(
        func="mj_char_events.reset_root_state",
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Joint state reset
    reset_joints: EventTermCfg = EventTermCfg(
        func="mj_char_events.reset_joint_state",
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# =============================================================================
# Action Configuration
# =============================================================================

@configclass
class ActionsCfg:
    """
    Action configuration for character control.

    Supports multiple control modes from ControlMode enum:
        - pd: Position targets with internal PD control
        - vel: Velocity targets
        - torque: Direct torque control
        - pd_exp: Custom exponential map PD control (for 3D joints)
        - pd_1d: Custom 1D PD control (for hinge joints)
    """

    # Default: PD position control targeting all joints
    joint_pos: Any = None  # Set based on control_mode at runtime


# =============================================================================
# Main Environment Configuration
# =============================================================================

@configclass
class MJCharEnvCfg(ManagerBasedRLEnvCfg):
    """
    Main configuration for the PARC character environment using mjlab.

    This is the mjlab equivalent of IGCharEnv configuration, structured
    using the Manager-Based RL API pattern.

    Usage:
        cfg = MJCharEnvCfg()
        cfg.scene.robot.spawn = MjcfFileCfg(asset_path="humanoid.xml")
        cfg.env.char_file = "path/to/humanoid.xml"
        env = ManagerBasedRLEnv(cfg=cfg)
    """

    # Scene configuration
    scene: MJCharSceneCfg = MJCharSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
    )

    # Observation configuration
    observations: ObservationsCfg = ObservationsCfg()

    # Action configuration
    actions: ActionsCfg = ActionsCfg()

    # Reward configuration
    rewards: RewardsCfg = RewardsCfg()

    # Termination configuration
    terminations: TerminationsCfg = TerminationsCfg()

    # Event configuration (resets)
    events: EventCfg = EventCfg()

    # ---------------------------
    # Environment-specific settings (ported from env_config in Isaac Gym)
    # ---------------------------

    # Character file path (MJCF XML)
    char_file: str = ""

    # Control mode
    control_mode: ControlMode = ControlMode.pd_exp

    # Episode settings
    episode_length_s: float = 10.0

    # Simulation settings
    decimation: int = 6  # sim_freq / control_freq = 60 / 10

    # Camera mode for visualization
    camera_mode: CameraMode = CameraMode.track

    # Global vs local observations
    global_obs: bool = False
    root_height_obs: bool = True

    # Motion library settings
    motion_file: str = ""
    use_contact_info: bool = False

    # Key bodies for observation/reward
    key_bodies: List[str] = field(default_factory=list)

    # Contact bodies for termination/reward
    contact_bodies: List[str] = field(default_factory=list)

    # DeepMimic settings
    enable_early_termination: bool = True
    termination_height: float = 0.3
    pose_termination: bool = False
    pose_termination_dist: float = 1.0
    root_pos_termination_dist: float = 1.0
    root_rot_termination_angle: float = 1.0

    # Target observation settings
    tar_obs_steps: List[int] = field(default_factory=lambda: [1])
    enable_tar_obs: bool = True

    # Reward weights (normalized internally)
    pose_w: float = 0.5
    vel_w: float = 0.1
    root_pos_w: float = 0.15
    root_vel_w: float = 0.1
    key_pos_w: float = 0.15

    # Track root position (xy) in reward
    track_root: bool = True
    track_root_h: bool = True

    # Reference character visualization offset
    ref_char_offset: Tuple[float, float, float] = (2.0, 0.0, 0.0)

    def __post_init__(self):
        """Post-initialization to set derived values."""
        # Set simulation timestep (60 Hz physics, 10 Hz control)
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        # Physics settings (similar to Isaac Gym PhysX)
        self.sim.gravity = (0.0, 0.0, -9.81)

        # Normalize reward weights
        total_w = self.pose_w + self.vel_w + self.root_pos_w + self.root_vel_w + self.key_pos_w
        if total_w > 0:
            self.rewards.pose.weight = self.pose_w / total_w
            self.rewards.vel.weight = self.vel_w / total_w
            self.rewards.root_pos.weight = self.root_pos_w / total_w
            self.rewards.root_vel.weight = self.root_vel_w / total_w
            self.rewards.key_pos.weight = self.key_pos_w / total_w

        # Update observation params
        self.observations.policy.root_rot.params["global_obs"] = self.global_obs
        self.observations.policy.root_lin_vel.params["global_obs"] = self.global_obs
        self.observations.policy.root_ang_vel.params["global_obs"] = self.global_obs
        self.observations.policy.key_body_pos.params["key_bodies"] = self.key_bodies
        self.observations.policy.key_body_pos.params["global_obs"] = self.global_obs

        # Update target observation params
        self.observations.target.tar_root_pos.params["tar_obs_steps"] = self.tar_obs_steps
        self.observations.target.tar_root_rot.params["tar_obs_steps"] = self.tar_obs_steps
        self.observations.target.tar_joint_rot.params["tar_obs_steps"] = self.tar_obs_steps
        self.observations.target.tar_key_body_pos.params["tar_obs_steps"] = self.tar_obs_steps
        self.observations.target.tar_key_body_pos.params["key_bodies"] = self.key_bodies

        # Update reward params
        self.rewards.root_pos.params["track_root_h"] = self.track_root_h
        self.rewards.key_pos.params["key_bodies"] = self.key_bodies

        # Update termination params
        self.terminations.root_height.params["termination_height"] = self.termination_height

        # Update event params
        self.events.reset_motion.params["motion_file"] = self.motion_file

        # Enable/disable optional observation groups
        if self.use_contact_info:
            self.observations.contacts = ContactObsCfg()
            self.observations.contacts.tar_contacts.params["tar_obs_steps"] = self.tar_obs_steps

        # Enable pose termination if configured
        if self.pose_termination:
            self.terminations.pose_deviation = TerminationTermCfg(
                func="mj_char_terminations.pose_deviation_termination",
                time_out=False,
                params={"pose_termination_dist": self.pose_termination_dist},
            )


# =============================================================================
# Parkour Environment Configuration (extends base)
# =============================================================================

@configclass
class MJParkourEnvCfg(MJCharEnvCfg):
    """
    Extended configuration for parkour/DeepMimic environment.

    This corresponds to IGParkourEnv configuration with additional
    settings for heightmap terrain and advanced motion tracking.
    """

    # Heightmap terrain settings
    use_heightmap: bool = True

    # Ray casting for local heightfield observation
    ray_points_behind: int = 3
    ray_points_ahead: int = 5
    ray_num_left: int = 3
    ray_num_right: int = 3
    ray_dx: float = 0.3
    ray_angle: float = 15.0

    # Contact detection threshold
    contact_detection_eps: float = 1e-5

    # Contact weights for reward computation
    contact_weights: List[float] = field(default_factory=list)

    # Fraction of environments using DeepMimic
    fraction_dm_envs: float = 1.0

    # Random reset from motion library
    rand_reset: bool = True

    # Demo mode settings
    demo_mode: bool = False
    never_done: bool = False

    # Tracking error reporting
    report_tracking_error: bool = False

    def __post_init__(self):
        """Post-initialization for parkour-specific settings."""
        super().__post_init__()

        # Enable heightfield observations
        if self.use_heightmap:
            self.observations.heightfield = HeightfieldObsCfg()
            self.observations.heightfield.local_heightfield.params.update({
                "ray_points_behind": self.ray_points_behind,
                "ray_points_ahead": self.ray_points_ahead,
                "ray_num_left": self.ray_num_left,
                "ray_num_right": self.ray_num_right,
                "ray_dx": self.ray_dx,
                "ray_angle": self.ray_angle,
            })

        # Enable contact penalty reward if contact info is used
        if self.use_contact_info and len(self.contact_weights) > 0:
            self.rewards.contact_penalty = RewardTermCfg(
                func="mj_char_rewards.contact_penalty",
                weight=1.0,
                params={"contact_weights": self.contact_weights},
            )
