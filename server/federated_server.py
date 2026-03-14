"""
server/federated_server.py  ─ v2
Federated server with:
  - Pluggable aggregation: FedAvg | FedProx | FedMedian | SCAFFOLD
  - SMPC / Secure Aggregation support
  - Full per-round logging for all metrics
"""

import torch
import torch.nn as nn
import copy
import os
import json
import logging
from typing import List, Dict, Optional

from server.aggregation_methods import get_aggregator, SCAFFOLDAggregator

logger = logging.getLogger(__name__)


class FederatedServer:
    def __init__(
        self,
        global_model:      nn.Module,
        dp_mechanism,
        device:            torch.device,
        results_dir:       str  = "results",
        aggregation_method:str  = "FedAvg",
        fedprox_mu:        float = 0.01,
        use_smpc:          bool  = False,
        smpc_aggregator    = None,
    ):
        self.global_model    = global_model.to(device)
        self.dp              = dp_mechanism
        self.device          = device
        self.results_dir     = results_dir
        self.round_history   = []
        self.best_acc        = 0.0
        self.method_name     = aggregation_method
        self.use_smpc        = use_smpc
        self.smpc            = smpc_aggregator

        # Build aggregator
        self.aggregator = get_aggregator(
            aggregation_method, device=str(device), mu=fedprox_mu
        )
        os.makedirs(results_dir, exist_ok=True)
        logger.info(
            f"Server: method={aggregation_method} | "
            f"SMPC={'enabled' if use_smpc else 'disabled'}"
        )

    def get_global_weights(self) -> Dict:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates: List[Dict],
                  selected_cids: Optional[List[str]] = None) -> Dict:
        """
        Run the selected aggregation method (FedAvg / FedProx / FedMedian / SCAFFOLD).
        Optionally use SMPC for secure aggregation instead of plaintext FedAvg.
        """
        if not client_updates:
            return self.get_global_weights()

        total   = sum(u["data_size"] for u in client_updates)
        weights = [u["data_size"] / total for u in client_updates]
        base    = self.get_global_weights()

        if self.use_smpc and self.smpc is not None:
            # ── SMPC path: server never sees individual updates ────────────────
            new_weights = self.smpc.secure_aggregate(client_updates, weights, base)
            logger.info(f"SMPC aggregation complete ({len(client_updates)} clients)")
        else:
            # ── Standard cryptographic aggregation path ────────────────────────
            kwargs = {}
            if isinstance(self.aggregator, SCAFFOLDAggregator):
                kwargs["selected_cids"] = selected_cids
            new_weights = self.aggregator.aggregate(
                client_updates, weights, base, **kwargs
            )

            # Optional server-side DP noise on the aggregate
            if self.dp.enabled:
                delta = {
                    k: (new_weights[k] - base[k]).float()
                    for k in new_weights if base[k].is_floating_point()
                }
                delta       = self.dp.privatize_model_update(delta)
                new_weights = {
                    k: (base[k] + delta[k]) if k in delta else base[k].clone()
                    for k in base
                }

        self.global_model.load_state_dict(new_weights)
        return self.get_global_weights()

    def log_round(self, round_num: int, selected_clients: List[str],
                  client_updates: List[Dict], global_acc: float,
                  rl_reward: float, rl_loss: float, privacy_eps: float) -> Dict:
        avg_loss = sum(u["train_loss"]   for u in client_updates) / len(client_updates)
        avg_acc  = sum(u["val_accuracy"] for u in client_updates) / len(client_updates)
        rec = {
            "round":            round_num,
            "method":           self.method_name,
            "selected_clients": selected_clients,
            "avg_local_loss":   round(avg_loss,    4),
            "avg_local_acc":    round(avg_acc,     4),
            "global_acc":       round(global_acc,  4),
            "rl_reward":        round(rl_reward,   4),
            "rl_loss":          round(rl_loss,     6),
            "privacy_epsilon":  round(privacy_eps, 4),
        }
        self.round_history.append(rec)
        logger.info(
            f"[Round {round_num:02d}][{self.method_name}] "
            f"GlobalAcc={global_acc:.4f} | Loss={avg_loss:.4f} | "
            f"RLReward={rl_reward:.4f} | eps={privacy_eps:.4f}"
        )
        if global_acc > self.best_acc:
            self.best_acc = global_acc
            try:
                import config
                if getattr(config, "SAVE_MODEL_CHECKPOINTS", False):
                    self.save_checkpoint("best_model.pt")
            except Exception:
                pass
            logger.info(f"  NEW BEST: {global_acc:.4f}")
        return rec

    def save_checkpoint(self, filename: str):
        torch.save(
            {"model_state_dict": self.global_model.state_dict(),
             "best_acc": self.best_acc, "method": self.method_name},
            os.path.join(self.results_dir, filename)
        )

    def save_history(self):
        path = os.path.join(self.results_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.round_history, f, indent=2)
        logger.info(f"History saved: {path}")
        return path
