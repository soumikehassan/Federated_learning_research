"""
server/aggregation_methods.py  ─ v2.1 (fixed)
Multiple FL aggregation methods for comparison.

FIX: Each client has a different classification head size (4, 7, 2 classes).
     Only the shared Swin backbone weights are aggregated.
     Head weights (head.fc.weight / head.fc.bias) are kept client-local.

Methods:
  1. FedAvg    — McMahan et al. (2017)
  2. FedProx   — Li et al. (2020)
  3. FedMedian — Yin et al. (2018) — robust to outliers
  4. SCAFFOLD  — Karimireddy et al. (2020) — fixes client drift
"""

import torch
import copy
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Helper: decide if a key belongs to the shared backbone ───────────────────
HEAD_KEYWORDS = ("head", "classifier", "fc", "linear")

def _is_backbone_key(key: str, updates: List[Dict]) -> bool:
    """
    Return True if this key can be safely averaged across all clients.
    A key is backbone-only if:
      1. It does NOT contain a head/classifier keyword, OR
      2. All clients have the exact same tensor shape for this key.
    """
    # Check if all clients agree on this key's shape
    shapes = set()
    for upd in updates:
        delta = upd.get("weight_delta", {})
        if key in delta:
            shapes.add(tuple(delta[key].shape))
    # If shapes differ → skip (it's a task-specific head layer)
    if len(shapes) > 1:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────────────────────
class BaseAggregator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def _backbone_keys(self, base: Dict, updates: List[Dict]) -> List[str]:
        """Return only the keys that are safe to aggregate (shared backbone)."""
        return [
            k for k in base
            if base[k].is_floating_point() and _is_backbone_key(k, updates)
        ]

    def aggregate(self, client_updates, data_weights, global_weights, **kwargs):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 1. FedAvg
# ─────────────────────────────────────────────────────────────────────────────
class FedAvgAggregator(BaseAggregator):
    """Weighted average of shared backbone weights only."""
    name = "FedAvg"

    def aggregate(self, client_updates, data_weights, global_weights, **kwargs):
        base = global_weights
        keys = self._backbone_keys(base, client_updates)

        delta = {k: torch.zeros_like(base[k], dtype=torch.float32) for k in keys}

        for upd, w in zip(client_updates, data_weights):
            for k in keys:
                if k in upd["weight_delta"]:
                    delta[k] += w * upd["weight_delta"][k].float().to(self.device)

        new_weights = {
            k: (base[k] + delta[k]) if k in delta else base[k].clone()
            for k in base
        }
        skipped = len(base) - len(keys)
        logger.info(f"FedAvg: {len(client_updates)} clients | "
                    f"{len(keys)} backbone keys aggregated | "
                    f"{skipped} head keys kept local")
        return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# 2. FedProx
# ─────────────────────────────────────────────────────────────────────────────
class FedProxAggregator(BaseAggregator):
    """FedAvg with proximal correction to reduce client drift."""
    name = "FedProx"

    def __init__(self, mu: float = 0.01, device: str = "cpu"):
        super().__init__(device)
        self.mu = mu

    def aggregate(self, client_updates, data_weights, global_weights, **kwargs):
        base = global_weights
        keys = self._backbone_keys(base, client_updates)

        delta = {k: torch.zeros_like(base[k], dtype=torch.float32) for k in keys}
        for upd, w in zip(client_updates, data_weights):
            for k in keys:
                if k in upd["weight_delta"]:
                    delta[k] += w * upd["weight_delta"][k].float().to(self.device)

        # Proximal shrinkage: reduces overfitting to local data
        new_weights = {
            k: (base[k] + delta[k] / (1.0 + self.mu)) if k in delta
            else base[k].clone()
            for k in base
        }
        logger.info(f"FedProx (mu={self.mu}): {len(client_updates)} clients aggregated")
        return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. FedMedian
# ─────────────────────────────────────────────────────────────────────────────
class FedMedianAggregator(BaseAggregator):
    """Coordinate-wise median — robust to outlier/poisoned clients."""
    name = "FedMedian"

    def aggregate(self, client_updates, data_weights, global_weights, **kwargs):
        base = global_weights
        keys = self._backbone_keys(base, client_updates)
        new_weights = {}

        for k in base:
            if k not in keys:
                new_weights[k] = base[k].clone()
                continue

            # Stack deltas from all clients → take element-wise median
            stacked = torch.stack([
                upd["weight_delta"][k].float().to(self.device)
                for upd in client_updates
                if k in upd["weight_delta"]
            ], dim=0)

            median_delta   = stacked.median(dim=0).values
            new_weights[k] = base[k] + median_delta

        logger.info(f"FedMedian: {len(client_updates)} clients | "
                    f"{len(keys)} backbone keys (robust aggregation)")
        return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCAFFOLD
# ─────────────────────────────────────────────────────────────────────────────
class SCAFFOLDAggregator(BaseAggregator):
    """
    Control variates to correct client drift on heterogeneous data.
    c_global tracks server gradient direction.
    c_clients[k] tracks each client's gradient direction.
    """
    name = "SCAFFOLD"

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.c_global  = None
        self.c_clients = {}
        self.lr_c      = 0.1

    def _init_controls(self, keys, base):
        self.c_global = {
            k: torch.zeros_like(base[k], dtype=torch.float32).to(self.device)
            for k in keys
        }

    def aggregate(self, client_updates, data_weights, global_weights,
                  selected_cids=None, **kwargs):
        base = global_weights
        keys = self._backbone_keys(base, client_updates)

        if self.c_global is None:
            self._init_controls(keys, base)

        # Init missing client control variates
        if selected_cids:
            for cid in selected_cids:
                if cid not in self.c_clients:
                    self.c_clients[cid] = {
                        k: torch.zeros_like(base[k], dtype=torch.float32).to(self.device)
                        for k in keys
                    }

        # Step 1: FedAvg delta
        delta = {k: torch.zeros_like(base[k], dtype=torch.float32) for k in keys}
        for upd, w in zip(client_updates, data_weights):
            for k in keys:
                if k in upd["weight_delta"]:
                    delta[k] += w * upd["weight_delta"][k].float().to(self.device)

        # Step 2: SCAFFOLD correction
        corrected = {}
        for k in keys:
            if (selected_cids and
                    all(cid in self.c_clients for cid in selected_cids)):
                avg_c = sum(
                    self.c_clients[cid][k] for cid in selected_cids
                ) / len(selected_cids)
                corrected[k] = delta[k] - avg_c + self.c_global[k]
            else:
                corrected[k] = delta[k]

        # Step 3: Update control variates
        for k in keys:
            self.c_global[k] = self.c_global[k] + self.lr_c * delta[k]
        if selected_cids:
            for cid, upd in zip(selected_cids, client_updates):
                if cid in self.c_clients:
                    for k in keys:
                        if k in upd["weight_delta"]:
                            self.c_clients[cid][k] = (
                                self.c_clients[cid][k]
                                + self.lr_c * upd["weight_delta"][k].float().to(self.device)
                            )

        new_weights = {
            k: (base[k] + corrected[k]) if k in corrected else base[k].clone()
            for k in base
        }
        logger.info(f"SCAFFOLD: {len(client_updates)} clients | drift corrected")
        return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def get_aggregator(method: str, device: str = "cpu", **kwargs) -> BaseAggregator:
    """
    Returns the requested aggregator instance.
    method: "FedAvg" | "FedProx" | "FedMedian" | "SCAFFOLD"
    """
    mapping = {
        "FedAvg":    FedAvgAggregator,
        "FedProx":   FedProxAggregator,
        "FedMedian": FedMedianAggregator,
        "SCAFFOLD":  SCAFFOLDAggregator,
    }
    if method not in mapping:
        raise ValueError(f"Unknown method: {method}. Choose: {list(mapping.keys())}")

    if method == "FedProx":
        return FedProxAggregator(mu=kwargs.get("mu", 0.01), device=device)
    return mapping[method](device=device)