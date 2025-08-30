# 1. Add these required imports
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Add this import for the Go2 asset
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

@configclass
class TestEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # Call the parent's post_init method first
        super().__post_init__()

        #-- Scene Configuration
        # Replace the Cartpole with the Go2 robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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

        #-- Action Configuration
        # Set the scale for joint position actions
        self.actions.joint_pos.scale = 0.25

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