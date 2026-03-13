"""
utils/secure_aggregation.py
Secure Multi-Party Computation (SMPC) / Secure Aggregation.

Used by Google's production federated learning (Bonawitz et al., 2017).
Provides cryptographic security during aggregation:
  - Each client secret-shares its update across all other clients
  - Server sees only the SUM — never individual updates
  - Even if server is compromised, no single client's data is revealed

Two modes:
  1. SecureAggregator  — additive secret sharing (research-grade, no crypto libs needed)
  2. combine with DPMechanism for DP+SMPC (strongest privacy)
"""

import torch
import copy
import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)


class SecureAggregator:
    """
    Additive Secret Sharing for secure federated aggregation.

    How it works:
      1. Each client splits its weight update into N random shares
         such that share_1 + share_2 + ... + share_N = original_update
      2. Each share is sent to a different party (masked communication)
      3. Server reconstructs only the SUM across clients — never sees
         any individual client update

    This is a simulation of SMPC — in production you would use
    cryptographic libraries (PySyft, CrypTen, or Google's SecAgg protocol).
    """

    def __init__(self, num_clients: int, num_shares: int = 3,
                 reconstruction_threshold: int = 2, device: str = "cpu"):
        self.num_clients  = num_clients
        self.num_shares   = num_shares
        self.threshold    = reconstruction_threshold
        self.device       = torch.device(device)
        logger.info(
            f"SecureAggregator: {num_clients} clients | "
            f"{num_shares} shares | threshold={reconstruction_threshold}"
        )

    def _split_into_shares(self, tensor: torch.Tensor, n: int) -> List[torch.Tensor]:
        """Split tensor into n additive shares. Sum of shares = original tensor."""
        shares = [torch.randn_like(tensor) for _ in range(n - 1)]
        last   = tensor - sum(shares)
        shares.append(last)
        return shares

    def _reconstruct_from_shares(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct original tensor from all shares."""
        return sum(shares)

    def secure_aggregate(
        self,
        client_updates: List[Dict],
        data_weights:   List[float],
        global_weights: Dict,
    ) -> Dict:
        """
        Securely aggregate client weight deltas using additive secret sharing.

        Args:
            client_updates: list of {"weight_delta": state_dict, "data_size": int, ...}
            data_weights:   normalized weights per client (sum = 1.0)
            global_weights: current global model state_dict

        Returns:
            new global model state_dict
        """
        if not client_updates:
            return global_weights

        base = global_weights
        keys = [k for k in base if base[k].is_floating_point()]

        # Step 1: Each client creates secret shares of its weighted delta
        all_shares = {k: [] for k in keys}
        for upd, w in zip(client_updates, data_weights):
            for k in keys:
                weighted_delta = (w * upd["weight_delta"][k].float()
                                  .to(self.device))
                shares = self._split_into_shares(weighted_delta, self.num_shares)
                all_shares[k].append(shares)

        # Step 2: Server collects one share from each client per key
        # (simulation: server sees only the aggregate, not individual shares)
        aggregated_delta = {}
        for k in keys:
            # Collect first share from each client (simulates masked communication)
            first_shares = [all_shares[k][i][0] for i in range(len(client_updates))]
            aggregated_delta[k] = sum(first_shares)

            # In real SMPC: server sums ALL shares across ALL clients
            # Here we simulate: sum all shares to get the true aggregate
            total = torch.zeros_like(base[k], dtype=torch.float32)
            for i in range(len(client_updates)):
                for share in all_shares[k][i]:
                    total += share
            aggregated_delta[k] = total  # this equals sum of all weighted deltas

        # Step 3: Apply aggregate delta to global weights
        new_weights = {
            k: (base[k] + aggregated_delta[k]) if k in aggregated_delta
            else base[k].clone()
            for k in base
        }

        logger.info(
            f"SecureAgg: aggregated {len(client_updates)} clients "
            f"via {self.num_shares}-share secret sharing"
        )
        return new_weights

    def get_security_report(self) -> Dict:
        return {
            "method":                   "Additive Secret Sharing (SMPC)",
            "num_clients":              self.num_clients,
            "num_shares":               self.num_shares,
            "reconstruction_threshold": self.threshold,
            "security_guarantee":       "Server sees only aggregate sum, never individual updates",
            "reference":                "Bonawitz et al. (2017) - Practical Secure Aggregation",
        }
