# FILE: .../custom_mdp/curriculums.py

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_progress_from_velocity_tracking(env: ManagerBasedRLEnv) -> float:
    """
    velocity tracking error 기반 커리큘럼 진행률(0.0 ~ 1.0)
    """
    # 목표 속도 (vx, vy, yaw_rate) from command manager
    commanded = env.command_manager.get_term("base_velocity").command  # (num_envs, 3)

    # 실제 속도 (scene에서 직접 가져오기)
    robot = env.scene["robot"]
    measured_lin = robot.data.root_lin_vel_b[:, :2]   # vx, vy
    measured_ang = robot.data.root_ang_vel_b[:, 2:3]  # yaw rate (z축)
    measured = torch.cat([measured_lin, measured_ang], dim=-1)

    # velocity tracking error
    error = torch.norm(commanded - measured, dim=-1)  # per-env error
    mean_error = error.mean()

    # 허용 가능한 최대 오차
    max_error = 1.0

    progress = 1.0 - (mean_error / max_error)
    return torch.clamp(progress, 0.0, 1.0).item()



def modify_command_ranges(
    env: ManagerBasedRLEnv,
    env_ids: list[int],
    ranges: dict[str, tuple[float, float]],
):
    """명령 범위를 점진적으로 넓힘 (velocity tracking 성능 기반)."""
    progress = _get_progress_from_velocity_tracking(env)

    command_term = env.command_manager.get_term("base_velocity")

    for vel_name, vel_range in ranges.items():
        new_min = vel_range[0] * progress
        new_max = vel_range[1] * progress
        setattr(command_term.cfg.ranges, vel_name, (new_min, new_max))


def curriculum_physics_material(
    env: ManagerBasedRLEnv,
    env_ids: list[int],
):
    """물리 재질의 랜덤화 범위를 점진적으로 조절 (velocity tracking 성능 기반)."""
    start_ranges = {
        "static_friction_range": (0.8, 0.8),
        "dynamic_friction_range": (0.6, 0.6),
        "restitution_range": (0.0, 0.0),
    }
    end_ranges = {
        "static_friction_range": (0.05, 4.5),
        "dynamic_friction_range": (0.05, 4.5),
        "restitution_range": (0.0, 1.0),
    }

    progress = _get_progress_from_velocity_tracking(env)

    event_term_name = "physics_material"
    event_cfg = env.event_manager.get_term_cfg(event_term_name)

    for param_name, end_range in end_ranges.items():
        start_range = start_ranges[param_name]
        new_min = start_range[0] + (end_range[0] - start_range[0]) * progress
        new_max = start_range[1] + (end_range[1] - start_range[1]) * progress
        event_cfg.params[param_name] = (new_min, new_max)



# # FILE: .../custom_mdp/curriculums.py

# from __future__ import annotations
# from typing import TYPE_CHECKING
# import torch

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv




# def _get_progress_from_episode_length(env: ManagerBasedRLEnv) -> float:
#     """
#     현재 환경들의 평균 에피소드 길이를 기반으로 커리큘럼 진행률(0.0 ~ 1.0)을 계산합니다.
#     에이전트가 평균적으로 에피소드 최대 길이의 80% 이상 생존하면 진행률이 1.0이 됩니다.
#     """
#     mean_episode_length = env.episode_length_buf.float().mean()
#     target_length = env.max_episode_length * 0.8
#     # 0으로 나누는 것을 방지
#     if target_length == 0:
#         return 1.0
#     progress = mean_episode_length / target_length
#     return torch.clamp(progress, 0.0, 1.0).item()


# def modify_command_ranges(
#     env: ManagerBasedRLEnv,
#     env_ids: list[int],
#     ranges: dict[str, tuple[float, float]],
# ):
#     """명령 범위를 점진적으로 넓힙니다."""
#     # [수정] 에피소드 길이 기반의 진행률 계산 방식을 사용합니다.
#     progress = _get_progress_from_episode_length(env
#                                                 #  , exponent=1.5
#                                                  )
    
#     command_term = env.command_manager.get_term("base_velocity")

#     for vel_name, vel_range in ranges.items():
#         new_min = vel_range[0] * progress
#         new_max = vel_range[1] * progress
#         setattr(command_term.cfg.ranges, vel_name, (new_min, new_max))


# def curriculum_physics_material(
#     env: ManagerBasedRLEnv,
#     env_ids: list[int],
# ):
#     """물리 재질의 랜덤화 범위를 조절합니다."""
    
#     start_ranges = {
#         "static_friction_range": (0.8, 0.8),
#         "dynamic_friction_range": (0.6, 0.6),
#         "restitution_range": (0.0, 0.0),
#     }
#     end_ranges = {
#         "static_friction_range": (0.05, 4.5),
#         "dynamic_friction_range": (0.05, 4.5),
#         "restitution_range": (0.0, 1.0),
#     }
    
#     # [수정] 에피소드 길이 기반의 진행률 계산 방식을 사용합니다.
#     progress = _get_progress_from_episode_length(env
#                                                 #  , exponent=1.0
#                                                  )

#     event_term_name = "physics_material"
#     event_cfg = env.event_manager.get_term_cfg(event_term_name)
    
#     for param_name, end_range in end_ranges.items():
#         start_range = start_ranges[param_name]
#         new_min = start_range[0] + (end_range[0] - start_range[0]) * progress
#         new_max = start_range[1] + (end_range[1] - start_range[1]) * progress
#         event_cfg.params[param_name] = (new_min, new_max)

