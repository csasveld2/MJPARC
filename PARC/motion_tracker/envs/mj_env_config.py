"""
mjlab Environment Configuration for PARC Motion Tracking.

This module defines the configuration for the PARC motion tracking environment
using mjlab's Manager-Based RL API (MuJoCo Warp backend).
"""

from __future__ import annotations

import mujoco

# mjlab imports
from mjlab.envs import ManagerBasedRlEnvCfg
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


def make_parc_tracking_env_cfg(
    char_file: str,
    motion_file: str | None = None,
    num_envs: int = 4096,
    episode_length_s: float = 10.0,
) -> ManagerBasedRlEnvCfg:
    """
    Create PARC motion tracking environment configuration.

    Args:
        char_file: Path to character MJCF XML file
        motion_file: Path to motion library file (optional, not yet supported)
        num_envs: Number of parallel environments
        episode_length_s: Episode length in seconds

    Returns:
        ManagerBasedRlEnvCfg: Environment configuration
    """

    # Create the MJCF spec loader function
    def load_humanoid_spec() -> mujoco.MjSpec:
        """Load the humanoid MJCF file."""
        return mujoco.MjSpec.from_file(char_file)

    # Define articulation with actuators for humanoid
    # This creates position-controlled actuators for all joints
    humanoid_articulation = EntityArticulationInfoCfg(
        actuators=(
            # Torso joints (abdomen)
            BuiltinPositionActuatorCfg(
                target_names_expr=("abdomen_.*",),
                stiffness=1000.0,
                damping=100.0,
                effort_limit=200.0,
            ),
            # Neck joints
            BuiltinPositionActuatorCfg(
                target_names_expr=("neck_.*",),
                stiffness=100.0,
                damping=10.0,
                effort_limit=50.0,
            ),
            # Shoulder joints
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_shoulder_.*",),
                stiffness=400.0,
                damping=40.0,
                effort_limit=100.0,
            ),
            # Elbow joints
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_elbow",),
                stiffness=300.0,
                damping=30.0,
                effort_limit=70.0,
            ),
            # Hip joints
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_hip_.*",),
                stiffness=500.0,
                damping=50.0,
                effort_limit=200.0,
            ),
            # Knee joints
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_knee",),
                stiffness=500.0,
                damping=50.0,
                effort_limit=150.0,
            ),
            # Ankle joints
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_ankle_.*",),
                stiffness=400.0,
                damping=40.0,
                effort_limit=90.0,
            ),
        ),
        soft_joint_pos_limit_factor=0.9,
    )

    # Observation terms using mjlab's built-in functions
    obs_terms = {
        "base_lin_vel": ObservationTermCfg(func=mjlab_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mjlab_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=mjlab_mdp.projected_gravity),
        "joint_pos": ObservationTermCfg(func=mjlab_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mjlab_mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=mjlab_mdp.last_action),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=obs_terms,
            concatenate_terms=True,
        ),
    }

    # Action configuration - control all actuators
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }

    # Event configuration
    events: dict[str, EventTermCfg] = {
        "reset_scene": EventTermCfg(
            func=mjlab_mdp.reset_scene_to_default,
            mode="reset",
        ),
    }

    # Simple reward: stay alive (not terminated)
    rewards: dict[str, RewardTermCfg] = {
        "is_alive": RewardTermCfg(
            func=mjlab_mdp.is_alive,
            weight=1.0,
        ),
    }

    # Termination configuration
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(
            func=mjlab_mdp.time_out,
            time_out=True,
        ),
        "root_height": TerminationTermCfg(
            func=mjlab_mdp.root_height_below_minimum,
            params={"minimum_height": 0.3},
        ),
    }

    # Create the environment configuration
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=5.0,
            terrain=TerrainImporterCfg(terrain_type="plane"),
            entities={
                "robot": EntityCfg(
                    spec_fn=load_humanoid_spec,
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 1.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                    articulation=humanoid_articulation,
                ),
            },
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
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
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        episode_length_s=episode_length_s,
        decimation=4,
    )

    return cfg
