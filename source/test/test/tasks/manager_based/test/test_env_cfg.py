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
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
# The asset configuration for the Unitree Go2 robot.
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

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
class RewardsCfg(BaseRewardsCfg):
    """
    Inherits from the base rewards class and customizes it for our task.
    This class defines the "default" structure and weights for our rewards.
    """
    
    # ==========================================================================
    # Chapter 7 Additions: New terms for gait and posture shaping.
    # ==========================================================================
    # Penalize the base dropping too low to prevent the "spider-gait".
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.30}
    )
    # Penalize legs splaying out to encourage a more natural, narrow stance.
    hip_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 3, 6, 9])}
    )
    
    # --- Overrides of Base Rewards ---
    # Modify: Increase the penalty on motor torques for better energy efficiency.
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)

    # Remove: The feet_air_time reward is complex and not needed for a basic flat-terrain task.
    feet_air_time = None


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
            lin_vel_x=(3.0, 3.0),  # Force a constant forward speed of 3.0 m/s
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0)
        ),
    )


#
# -- Main Environment Configuration --
#

@configclass
class TestEnvCfg(LocomotionVelocityRoughEnvCfg):
    """The main configuration for our Go2 locomotion tutorial experiment."""

    # Override the base environment's actions and rewards with our custom classes.
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg() # Activate the curriculum

    def __post_init__(self):
        """Post-initialization to override specific parameters."""
        # Always call the parent's post_init first.
        super().__post_init__()

        # -- Scene Settings --
        # Replace the default robot with our Unitree Go2 asset.
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Simplify the environment to a flat plane for focused, faster training.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # With a flat plane, the height scanner is no longer needed.
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        # We don't need a curriculum for terrain difficulty on a flat plane.
        self.curriculum.terrain_levels = None

        # -- Event Settings --
        # Randomize the robot's starting pose on each reset to improve robustness.
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},  # Start from a standstill.
        }
        # Always reset joints to their default stable pose.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        
        # -- [WORKFLOW] Experiment-specific Reward Weight Overrides --
        # This section acts as a "control panel" to quickly tune weights for this
        # specific experiment, overriding the defaults set in the RewardsCfg class above.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # Note: We are overriding the dof_torques_l2 weight again here.
        # This takes final precedence.
        self.rewards.dof_torques_l2.weight = -0.0002
        # We explicitly set undesired_contacts to None to ensure it's disabled.
        self.rewards.undesired_contacts = None
        # We override the base_height_l2 weight from our class default for this run.
        self.rewards.base_height_l2.weight = -2.0

        # -- Termination Settings --
        # End the episode if the robot's base/torso touches the ground.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


#
# -- Evaluation-time Configuration --
#

@configclass
class TestEnvCfg_PLAY(TestEnvCfg):
    """Configuration for playing and evaluating a trained policy."""
    # Override the commands with our high-speed test version.
    commands: CommandsCfg_PLAY = CommandsCfg_PLAY()

    def __post_init__(self):
        super().__post_init__()
        # Disable the curriculum during play mode.
        self.curriculum = None

        # --- Settings to override for playing ---
        # Use a smaller number of environments for real-time visualization.
        self.scene.num_envs = 50
        # Disable all randomization for deterministic and consistent evaluation.
        self.observations.policy.enable_corruption = False
        # Remove random pushing events during play.
        self.events.base_external_force_torque = None
        self.events.push_robot = None