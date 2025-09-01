from isaaclab.utils import configclass
# Import the config class from the source code we verified
from isaaclab.actuators import ActuatorNetMLPCfg

@configclass
class Go1ActuatorCfg:
    """
    Configuration for the Unitree Go1's ActuatorNet model.
    This will replace the default implicit PD controller.
    """
    # Define a single group for all 12 leg joints
    base_legs = ActuatorNetMLPCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        # The path to the pre-trained neural network file
        network_file="/home/yoda/Projects/3d_locomotion_base_backup/genesis_lr/resources/actuator_nets/unitree_go1.pt",
        # Hardware specs (can be tuned for Go2)
        effort_limit=23.7,
        saturation_effort=23.7,
        velocity_limit=30.0,
        # Network-specific parameters, defined by the model's training
        input_idx=[0, 1, 2],
        input_order="pos_vel",
        pos_scale=-1.0,
        vel_scale=1.0,
        torque_scale=1.0,
    )

# For convenience, we create an instance to be easily imported
GO1_ACTUATORS_CFG = Go1ActuatorCfg()