# /home/yoda/Projects/personal_tutorial_isaaclab/test/custom_rsl/custom_ppo.py
from rsl_rl.algorithms import PPO
# CustomRolloutStorage를 import 합니다.
from .custom_rollout_storage import CustomRolloutStorage

class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[CUSTOM] Using Custom PPO!")
    
    # 이 메서드를 추가해야 합니다!
    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        print("[CUSTOM] PPO is initializing CUSTOM RolloutStorage!")
        # 기존 RolloutStorage 대신 CustomRolloutStorage를 사용하도록 합니다.
        self.storage = CustomRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            # rnd_state_shape 등 다른 인자가 필요하면 원본 코드를 참고하여 추가해야 합니다.
            # 지금은 기본 인자만 전달합니다.
            device=self.device,
        )

    def update(self):
        print("[CUSTOM] PPO update called")
        result = super().update()
        # 여기에 커스텀 로직 추가
        return result