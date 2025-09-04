# /home/yoda/Projects/personal_tutorial_isaaclab/test/custom_rsl/custom_on_policy_runner.py

from rsl_rl.runners import OnPolicyRunner
# CustomRolloutStorage를 여기서 import할 필요가 없습니다.

class CustomOnPolicyRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        # super().__init__()을 호출하는 것만으로도
        # 내부적으로 CustomPPO -> CustomRolloutStorage 생성까지 모두 처리됩니다.
        super().__init__(*args, **kwargs)
        print("[CUSTOM] Using Custom OnPolicyRunner!")
        # 여기서 self.storage를 덮어쓰는 코드는 필요 없습니다.
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        print("[CUSTOM] OnPolicyRunner learn called")
        return super().learn(num_learning_iterations, init_at_random_ep_len)