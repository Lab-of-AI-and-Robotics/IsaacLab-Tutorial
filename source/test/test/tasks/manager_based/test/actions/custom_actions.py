# FILE: .../test/source/test/test/tasks/manager_based/test/actions/custom_actions.py

import torch
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv

# --- This is the key part: we import our validated solvers ---
from ..kinematics.solver import go2_fk, go2_ik, HIP_OFFSETS

@configclass
class FootSpaceIKActionCfg(ActionTermCfg):
    """Configuration for the FootSpaceIKAction term."""
    class_type: type = MISSING
    # scale: float = 0.1
    scale: float = 0.0

class FootSpaceIKAction(ActionTerm):
    cfg: FootSpaceIKActionCfg

    def __init__(self, cfg: FootSpaceIKActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self._asset = self.env.scene[cfg.asset_name]
        
        # --- [KEY FIX] Joint Order Mapping Tensors ---
        # This mapping assumes our Kinematics solvers use (FR, FL, RR, RL) order,
        # while Isaac Sim's default order might be different.
        self.isaac_to_kinematics_map = torch.tensor([1, 0, 3, 2], device=self.device)
        self.kinematics_to_isaac_map = torch.tensor([1, 0, 3, 2], device=self.device)

        # Expand hip offsets for batch operations
        self.hip_offsets_batch = HIP_OFFSETS.expand(self.env.num_envs, -1, -1).to(self.device)
        
        # --- [KEY FIX] Correct Default Foot Position Calculation ---
        # 1. Get default joint positions in Isaac order
        default_q_isaac = self._asset.data.default_joint_pos.view(self.env.num_envs, 4, 3)
        # 2. Re-order them to match our kinematics solver's expectation
        default_q_kinematics = default_q_isaac[:, self.isaac_to_kinematics_map, :]
        
        # 3. Run FK with correctly ordered joints
        default_foot_pos_kinematics = go2_fk(
            default_q_kinematics.view(self.env.num_envs, 12),
            self.hip_offsets_batch
        )
        self.default_foot_positions = default_foot_pos_kinematics.view(self.env.num_envs, 4, 3)

        # Buffers
        self._raw_actions = torch.zeros(self.env.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.env.num_envs, self._asset.num_joints, device=self.device)

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
        self._raw_actions[:] = actions
        
        delta_foot_pos = actions.view(-1, 4, 3) * self.cfg.scale
        target_foot_positions = self.default_foot_positions + delta_foot_pos

        # IK solver gives us joint angles in our Kinematics order
        target_q_kinematics = go2_ik(target_foot_positions, self.hip_offsets_batch).view(self.env.num_envs, 4, 3)
        
        # --- [KEY FIX] Re-order the output back to Isaac order before sending to sim ---
        target_q_isaac = target_q_kinematics[:, self.kinematics_to_isaac_map, :]
        
        self._processed_actions[:] = target_q_isaac.view(self.env.num_envs, 12)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions)