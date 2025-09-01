# Copyright (c) 2022-2025, The Isaac Lab Project Developers. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ==== Tutorial-specific Imports ====
# Chapter 6: Import the custom action controller we created.
from .actions.custom_actions import FootSpaceIKAction, FootSpaceIKActionCfg

# ==== Isaac Lab Foundation Imports ====
# Utility for creating configuration classes.
from isaaclab.utils import configclass
# The base environment class we are inheriting from.
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

# ==== Isaac Lab Manager & MDP Imports ====
# Import various configuration objects for defining the environment.
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
# Import the library of standard reward and MDP functions.
import isaaclab.envs.mdp as mdp
# Chapter 7: Import the base RewardsCfg to inherit from it.
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg as BaseRewardsCfg
# Chapter 8: Curriculum Learning
# Import the helpers from the new file
from .curriculums import curriculum_helpers
# Import the CurriculumTermCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip



#
# -- Custom Action Configuration (from Chapter 6)
#

@configclass
class ActionsCfg:
    """Defines the custom action model for the Go2 robot."""
    # Action term for the custom foot-space IK controller.
    foot_ik = FootSpaceIKActionCfg(asset_name="robot", scale=0.05)
    # Explicitly link the config to its implementation class.
    foot_ik.class_type = FootSpaceIKAction


#
# -- Custom Reward Configuration (from Chapter 7)
#

@configclass
class H1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )



@configclass
class CurriculumCfg:
    """Defines the curriculum by calling the adaptive function from the helpers file."""
    command_velocity = CurrTerm(
        func=curriculum_helpers.modify_command_ranges,
        params={
            # We only need to provide the FINAL target ranges.
            "ranges": {
                "lin_vel_x": (-2.0, 2.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-1.0, 1.0)
            }
        },
    )

    # # [수정] params를 완전히 비워서 파라미터 불일치 에러를 방지합니다.
    # physics_material = CurrTerm(
    #     func=curriculum_helpers.curriculum_physics_material,
    #     params={},
    # )


@configclass
class CommandsCfg_PLAY:
    """A special command configuration to test the policy's limits at 3.0 m/s."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9), # Effectively never resample
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),  # Force a constant forward speed of 3.0 m/s
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0)
        ),
    )


#
# -- Main Environment Configuration --
#

@configclass
class TestEnvCfg(LocomotionVelocityRoughEnvCfg):
    """The main configuration for our H1 locomotion tutorial experiment."""

    rewards: H1Rewards = H1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"


@configclass
class TestEnvCfg_PLAY(TestEnvCfg):
    """Configuration for playing and evaluating a trained policy."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None