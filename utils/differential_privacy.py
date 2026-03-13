"""
utils/differential_privacy.py
Differential Privacy via Gaussian Mechanism.
Clips gradients + adds noise before server aggregation.
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class DPMechanism:
    def __init__(self, noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5, enabled=True):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm    = max_grad_norm
        self.delta            = delta
        self.enabled          = enabled
        self._steps           = 0
        logger.info(f"DP: enabled={enabled}, noise={noise_multiplier}, clip={max_grad_norm}")

    def clip_gradients(self, model: nn.Module):
        if self.enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    def add_noise_to_gradients(self, model: nn.Module):
        if not self.enabled:
            return
        std = self.noise_multiplier * self.max_grad_norm
        for p in model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * std)
        self._steps += 1

    def privatize_model_update(self, update: dict) -> dict:
        if not self.enabled:
            return update
        std = self.noise_multiplier * self.max_grad_norm
        return {
            k: (v + torch.randn_like(v) * std) if v.is_floating_point() else v.clone()
            for k, v in update.items()
        }

    def compute_epsilon(self, num_steps: int, sample_rate: float) -> float:
        if not self.enabled or num_steps == 0:
            return float("inf")
        try:
            return (
                math.sqrt(2 * math.log(1.25 / self.delta))
                / self.noise_multiplier
            ) * math.sqrt(num_steps * sample_rate)
        except Exception:
            return float("inf")

    def get_privacy_report(self, num_steps: int, sample_rate: float) -> dict:
        return {
            "epsilon":         self.compute_epsilon(num_steps, sample_rate),
            "delta":           self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm":   self.max_grad_norm,
            "steps":           num_steps,
        }
