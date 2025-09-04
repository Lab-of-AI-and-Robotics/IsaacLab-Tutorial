# /home/yoda/Projects/personal_tutorial_isaaclab/test/custom_rsl/custom_actor_critic.py
from rsl_rl.modules import ActorCritic

class CustomActorCritic(ActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[CUSTOM] Using Custom ActorCritic!")
    
    # 1. 메서드 정의에 **kwargs를 추가해서 예상치 못한 인자를 모두 받도록 합니다.
    def act(self, obs, privileged_obs=None, **kwargs):
        print("[CUSTOM] ActorCritic act called")
        # 2. 부모의 act를 호출할 때도 받은 kwargs를 그대로 전달해줍니다.
        return super().act(obs, **kwargs)