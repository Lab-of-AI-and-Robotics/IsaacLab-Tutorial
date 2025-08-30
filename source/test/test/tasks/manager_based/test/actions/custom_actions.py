# FILE: .../test/source/test/test/tasks/manager_based/test/actions/custom_actions.py

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

# Import our validated solver from the previous chapter
from ..kinematics.solver import go2_ik, HIP_OFFSETS, L_HIP, L_THIGH, L_CALF

@configclass
class FootSpaceIKActionCfg(ActionTermCfg):
    """Configuration for the FootSpaceIKAction term."""
    scale: float = 0.1  # Scale the policy's output delta positions

class FootSpaceIKAction(ActionTerm):
    """
    An action term that converts desired foot positions (task space) from the policy
    into target joint positions (joint space) using an IK solver.
    """
    cfg: FootSpaceIKActionCfg

    def __init__(self, cfg: FootSpaceIKActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Add this line to store the environment object
        self.env = env
        
        self._asset = env.scene[self.cfg.asset_name]

        # Buffers for actions
        # This will need to be changed to match the number of joints in the asset
        self._raw_actions = torch.zeros(self.env.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.env.num_envs, self._asset.num_joints, device=self.device)

        # We need the robot's default foot positions to use as a reference
        # We calculate this once at initialization using our FK solver
        # Note: This part requires the FK solver to be in solver.py as well
        # from ..kinematics.solver import go2_fk
        # default_q = self._asset.data.default_joint_pos
        # default_foot_pos = go2_fk(default_q, HIP_OFFSETS.unsqueeze(0).expand(self.env.num_envs, -1, -1).to(self.device))
        # self.default_foot_positions = default_foot_pos.view(self.env.num_envs, 4, 3)

        # For simplicity in this tutorial, we'll use a hardcoded default height
        self.default_foot_positions = torch.zeros(self.env.num_envs, 4, 3, device=self.device)
        self.default_foot_positions[..., 2] = -0.25 # Default height of -25cm

    @property
    def action_dim(self) -> int:
        return 12  # 4 legs * 3 (dx, dy, dz)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Processes the raw actions from the policy."""
        self._raw_actions[:] = actions
        
        # 1. Interpret policy output as deltas from the default foot position
        delta_foot_pos = actions.view(-1, 4, 3) * self.cfg.scale
        target_foot_positions = self.default_foot_positions + delta_foot_pos

        # 2. Call our IK solver to get the target joint angles
        target_joint_angles = go2_ik(
            target_foot_positions, 
            HIP_OFFSETS.expand(self.env.num_envs, -1, -1).to(self.device)
        )
        
        # 3. Store the result
        self._processed_actions[:] = target_joint_angles

    def apply_actions(self):
        """Applies the processed actions to the simulation."""
        # Set the joint position targets on the robot asset
        self._asset.set_joint_position_target(self._processed_actions)