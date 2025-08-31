# FILE: .../test/source/test/test/tasks/manager_based/test/actions/custom_actions.py

import torch
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv

# Import our validated solvers from the previous chapter
from ..kinematics.solver import go2_fk, go2_ik, HIP_OFFSETS

@configclass
class FootSpaceIKActionCfg(ActionTermCfg):
    """Configuration for the FootSpaceIKAction term."""
    class_type: type = MISSING
    scale: float = 0.1

class FootSpaceIKAction(ActionTerm):
    """
    A robust action term that correctly handles Isaac Lab's joint order and uses a
    dynamic reference point for stability.
    """
    cfg: FootSpaceIKActionCfg

    def __init__(self, cfg: FootSpaceIKActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # --- Finalized Initialization ---
        # Leg order mapping: sim (FL,FR,RL,RR) -> kinematics (FR,FL,RR,RL)
        self.kinematics_leg_order = torch.tensor([1, 0, 3, 2], device=self.device)
        # Leg order mapping: kinematics -> sim
        self.sim_leg_order = torch.tensor([1, 0, 3, 2], device=self.device)

        # Expand hip offsets for batch operations
        self.hip_offsets_batch = HIP_OFFSETS.expand(self._env.num_envs, -1, -1).to(self.device)
        
        # Buffers
        self._raw_actions = torch.zeros(self._env.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self._env.num_envs, self._asset.num_joints, device=self.device)

        # Add a one-time flag for debugging
        self._has_printed_debug_info = False

    @property
    def action_dim(self) -> int:
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        
        # --- Data pipeline with correct reshaping ---
        # 1. Get current joint positions from sim (shape: N, 12) (grouped by joint type)
        current_q_grouped = self._asset.data.joint_pos.clone()
        
        # 2. Reshape to (N, 3 joints, 4 legs) and transpose to (N, 4 legs, 3 joints)
        current_q_interleaved = current_q_grouped.view(-1, 3, 4).transpose(1, 2)
        
        # 3. Reorder legs to match our kinematics solver's expected order (FR, FL, RR, RL)
        current_q_kinematics = current_q_interleaved[:, self.kinematics_leg_order, :]

        # 4. Use FK to find the CURRENT foot positions (our dynamic reference)
        current_foot_pos = go2_fk(current_q_kinematics.view(-1, 12), self.hip_offsets_batch)
        
        # 5. Calculate target foot positions
        delta_foot_pos = actions.view(-1, 4, 3) * self.cfg.scale
        target_foot_pos = current_foot_pos.view(-1, 4, 3) + delta_foot_pos
        
        # 6. Run IK solver
        target_q_kinematics = go2_ik(target_foot_pos, self.hip_offsets_batch).view(-1, 4, 3)

        # 7. Reverse the process: reorder legs back to sim order (FL, FR, RL, RR)
        target_q_interleaved = target_q_kinematics[:, self.sim_leg_order, :]
        
        # 8. Transpose and reshape back to grouped-by-joint-type format (N, 12)
        target_q_grouped = target_q_interleaved.transpose(1, 2).reshape(-1, 12)
        
        self._processed_actions[:] = target_q_grouped

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions)