# /home/yoda/Projects/personal_tutorial_isaaclab/test/custom_rsl/custom_rollout_storage.py
from rsl_rl.storage import RolloutStorage

class CustomRolloutStorage(RolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[CUSTOM] Using Custom RolloutStorage!")
    
    def add(self, *args, **kwargs):
        print("[CUSTOM] RolloutStorage add called")
        return super().add(*args, **kwargs)
    
    def compute_returns(self, *args, **kwargs):
        print("[CUSTOM] RolloutStorage compute_returns called")
        return super().compute_returns(*args, **kwargs)