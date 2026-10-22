"""Measure learner collection, inference, and optimization without changing PPO."""

import torch
from sb3_contrib import RecurrentPPO

from smg_rl.diagnostics import StageProfiler, measured


class TimedRecurrentPPO(RecurrentPPO):
    @measured("rollout_collection")
    def collect_rollouts(self, *args, **kwargs):
        return super().collect_rollouts(*args, **kwargs)

    @measured("ppo_update")
    def train(self):
        # One synchronization per optimization boundary gives wall time rather
        # than just CUDA enqueue time, without synchronizing every operator.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result = super().train()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return result

    def install_inference_timing(self):
        if not hasattr(self, "profiler"):
            self.profiler = StageProfiler()
        if getattr(self, "_inference_timer_installed", False):
            return
        original = self.policy.forward

        def forward(*args, **kwargs):
            with self.profiler.measure("policy_forward"):
                return original(*args, **kwargs)

        self.policy.forward = forward
        self._inference_timer_installed = True

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["profiler", "_inference_timer_installed"]
