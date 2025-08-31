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
    scale: float = 0.1

class FootSpaceIKAction(ActionTerm):
    cfg: FootSpaceIKActionCfg

    def __init__(self, cfg: FootSpaceIKActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env
        self._asset = self._env.scene[cfg.asset_name]
        
        # This mapping assumes our Kinematics solvers use (FR, FL, RR, RL) order,
        # while Isaac Sim's default order might be different.
        self.isaac_to_kinematics_map = torch.tensor([1, 0, 3, 2], device=self.device)
        self.kinematics_to_isaac_map = torch.tensor([1, 0, 3, 2], device=self.device)

        # Expand hip offsets for batch operations
        self.hip_offsets_batch = HIP_OFFSETS.expand(self._env.num_envs, -1, -1).to(self.device)
        
        # 1. Get default joint positions in Isaac order
        default_q_isaac = self._asset.data.default_joint_pos.view(self._env.num_envs, 4, 3)
        # 2. Re-order them to match our kinematics solver's expectation
        default_q_kinematics = default_q_isaac[:, self.isaac_to_kinematics_map, :]
        
        # 3. Run FK with correctly ordered joints
        default_foot_pos_kinematics = go2_fk(
            default_q_kinematics.view(self._env.num_envs, 12),
            self.hip_offsets_batch
        )
        self.default_foot_positions = default_foot_pos_kinematics.view(self._env.num_envs, 4, 3)

        # Buffers
        self._raw_actions = torch.zeros(self._env.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self._env.num_envs, self._asset.num_joints, device=self.device)

        # Add a one-time flag for debugging
        self._has_printed_debug_info = False

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
        
        # 1. Get the CURRENT joint positions from the simulator
        current_q_isaac = self._asset.data.joint_pos.clone()
        
        # 2. Convert to our kinematics order
        current_q_kinematics = current_q_isaac.view(-1, 4, 3)[:, self.isaac_to_kinematics_map, :]
        
        # 3. Run FK to find the CURRENT foot positions
        current_foot_positions = go2_fk(current_q_kinematics.view(-1, 12), self.hip_offsets_batch)
        current_foot_positions = current_foot_positions.view(self._env.num_envs, 4, 3)
        
        # 4. Add the policy's delta to the CURRENT position to get the target
        delta_foot_pos = actions.view(-1, 4, 3) * self.cfg.scale
        target_foot_positions = current_foot_positions + delta_foot_pos
        
        # 5. Run IK to get target joint angles (in kinematics order)
        target_q_kinematics = go2_ik(target_foot_positions, self.hip_offsets_batch) # This is now calculated before the 'if' block

        # 6. Re-order the output back to Isaac order for the simulator
        target_q_isaac = target_q_kinematics.view(self._env.num_envs, 4, 3)[:, self.kinematics_to_isaac_map, :]
        
        self._processed_actions[:] = target_q_isaac.view(self._env.num_envs, 12)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions)