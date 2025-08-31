# FILE: .../test/source/test/test/tasks/manager_based/test/kinematics/solver.py

import torch

# --- Configuration Constants ---
L_HIP = 0.0955
L_THIGH = 0.213
L_CALF = 0.213
# Hip offsets in user-friendly order: FR, FL, RR, RL
HIP_OFFSETS = torch.tensor([
    [ 0.1934, -0.0465,  0.0 ],  # FR
    [ 0.1934,  0.0465,  0.0 ],  # FL
    [-0.1934, -0.0465,  0.0 ],  # RR
    [-0.1934,  0.0465,  0.0 ],  # RL
], dtype=torch.float64).unsqueeze(0)

# --- Kinematics Solvers ---
def go2_fk(joint_angles: torch.Tensor, hip_offsets: torch.Tensor) -> torch.Tensor:
    """Calculates foot positions in the base frame from joint angles."""
    batch_size = joint_angles.size(0)
    q = joint_angles.view(batch_size, 4, 3)
    q_hip, q_thigh, q_calf = q[:, :, 0], q[:, :, 1], q[:, :, 2]
    
    # Position of the foot in the leg's sagittal (2D) plane
    cos_q_thigh, sin_q_thigh = torch.cos(q_thigh), torch.sin(q_thigh)
    cos_q_thigh_calf, sin_q_thigh_calf = torch.cos(q_thigh + q_calf), torch.sin(q_thigh + q_calf)
    x_leg = L_THIGH * sin_q_thigh + L_CALF * sin_q_thigh_calf
    z_leg = -L_THIGH * cos_q_thigh - L_CALF * cos_q_thigh_calf

    # Apply 3D rotation for the hip joint (rotation around X-axis)
    side_sign = torch.tensor([-1, 1, -1, 1], device=joint_angles.device, dtype=torch.float64).unsqueeze(0)
    y_leg_frame = side_sign * L_HIP
    cos_q_hip, sin_q_hip = torch.cos(q_hip), torch.sin(q_hip)
    x_hip_frame = x_leg
    y_hip_frame = y_leg_frame * cos_q_hip - z_leg * sin_q_hip
    z_hip_frame = y_leg_frame * sin_q_hip + z_leg * cos_q_hip
    
    # Combine with hip offsets to get the final position in the base frame
    foot_pos_base_frame = torch.stack([x_hip_frame, y_hip_frame, z_hip_frame], dim=-1) + hip_offsets
    return foot_pos_base_frame.view(batch_size, 12)

def go2_ik(target_foot_positions: torch.Tensor, hip_offsets: torch.Tensor) -> torch.Tensor:
    """Calculates joint angles from target foot positions in the base frame."""
    batch_size = target_foot_positions.size(0)
    foot_targets_hip_frame = target_foot_positions - hip_offsets
    x, y, z = foot_targets_hip_frame[..., 0], foot_targets_hip_frame[..., 1], foot_targets_hip_frame[..., 2]

    # --- Hip Angle (q_hip) ---
    side_sign = torch.tensor([-1, 1, -1, 1], device=target_foot_positions.device, dtype=torch.float64).unsqueeze(0)
    y_offset = side_sign * L_HIP
    len_yz_sq = y**2 + z**2
    z_leg = -torch.sqrt(torch.clamp(len_yz_sq - y_offset**2, min=1e-8))
    sin_q_hip = (z * y_offset - y * z_leg) / len_yz_sq
    cos_q_hip = (y * y_offset + z * z_leg) / len_yz_sq
    hip_angles = torch.atan2(sin_q_hip, cos_q_hip)
    
    # --- Thigh and Calf Angles (q_thigh, q_calf) ---
    x_leg = x
    l_leg_sq = x_leg**2 + z_leg**2
    cos_calf = (l_leg_sq - L_THIGH**2 - L_CALF**2) / (2 * L_THIGH * L_CALF)
    cos_calf = torch.clamp(cos_calf, -1.0, 1.0)
    calf_angles = torch.acos(cos_calf) * -1.0
    
    k1 = L_THIGH + L_CALF * torch.cos(calf_angles)
    k2 = L_CALF * torch.sin(calf_angles)
    thigh_angles = torch.atan2(x_leg, -z_leg) - torch.atan2(k2, k1)

    joint_angles = torch.stack((hip_angles, thigh_angles, calf_angles), dim=-1)
    return joint_angles.view(batch_size, -1)

# --- Round-trip Test ---

def run_kinematics_round_trip_test():
    """Defines a test case and validates the FK and IK solvers."""
    print("="*50)
    print("Go2 Kinematics Round-trip Test")
    print("="*50)
    
    # 1. Define initial joint angles
    q_initial = torch.tensor([[ -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5, ]], dtype=torch.float64)
    print(f"Initial Joint Angles (rad):\n{q_initial.numpy().round(4)}")

    # 2. FK: Calculate foot positions
    foot_positions = go2_fk(q_initial, HIP_OFFSETS)
    print(f"\nCalculated Foot Positions (m):\n{foot_positions.view(1,4,3).numpy().round(4)}")

    # 3. IK: Recalculate joint angles from the foot positions
    q_recalculated = go2_ik(foot_positions.view(1, 4, 3), HIP_OFFSETS)
    print(f"\nRecalculated Joint Angles (rad):\n{q_recalculated.numpy().round(4)}")

    # 4. Verify the error is negligible
    error = torch.abs(q_initial - q_recalculated)
    print(f"\nMax Absolute Error: {torch.max(error).item():.2e}")
    assert torch.all(error < 1e-6), "Test Failed!"
    print("\n✅ Kinematics solver passed the round-trip test!")
    print("="*50)

if __name__ == "__main__":
    run_kinematics_round_trip_test()