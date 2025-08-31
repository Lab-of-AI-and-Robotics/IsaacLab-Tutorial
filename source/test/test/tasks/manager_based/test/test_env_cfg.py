# Add this import at the top
from .actions.custom_actions import FootSpaceIKAction, FootSpaceIKActionCfg

# 1. Add these required imports
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Add this import for the Go2 asset
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# Add EventTermCfg to the list of imports from isaaclab.managers
from isaaclab.managers import EventTermCfg
import isaaclab.envs.mdp as mdp

# Find and replace the existing ActionsCfg class with this
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Create the config object
    foot_ik = FootSpaceIKActionCfg(asset_name="robot", scale=0.05)
    # Explicitly assign the implementation class to the 'class_type' field
    foot_ik.class_type = FootSpaceIKAction


@configclass
class TestEnvCfg(LocomotionVelocityRoughEnvCfg):
    # This single line is the critical switch that activates our custom controller.
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        # 1. Call the parent's post_init method first
        super().__post_init__()

        #-- Scene Configuration
        # Replace the Cartpole with the Go2 robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set the initial joint positions to the stable pose
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Change terrain to a flat plane for faster training
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove the height scanner since the terrain is flat
        self.scene.height_scanner = None
        # Also remove height scan from observations
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        # Disable the terrain curriculum
        self.curriculum.terrain_levels = None

        # #-- Action Configuration
        # # Set the scale for joint position actions
        # self.actions.joint_pos.scale = 0.25

        #-- Event Configuration
        # Configure the robot's initial state randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},  # Default to zero velocity
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        
        #-- Reward Configuration
        # These weights are crucial for learning to walk!
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None # From previous fix
        # self.rewards.base_height_l2.weight = -1.0
        # self.rewards.flat_orientation_l2.weight = -0.5

        #-- Termination Configuration
        # The episode ends if the robot's base hits the ground
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class TestEnvCfg_PLAY(TestEnvCfg):
    def __post_init__(self):
        # First, inherit all settings from the training config
        super().__post_init__()


        # --- Settings to override for playing ---
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None