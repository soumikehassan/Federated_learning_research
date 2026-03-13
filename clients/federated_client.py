"""
clients/federated_client.py  ─ v2.2 (fixed)
FIX: set_model_weights() now skips head keys that don't match the
     local model shape — so each client keeps its own classification head
     while receiving the shared backbone from the server.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FederatedClient:
    def __init__(
        self,
        client_id:    str,
        model:        nn.Module,
        dataloaders:  Dict,
        dp_mechanism,
        device:       torch.device,
        lr:           float = 1e-4,
        weight_decay: float = 1e-4,
        local_epochs: int   = 3,
        fedprox_mu:   float = 0.0,
    ):
        self.client_id    = client_id
        self.model        = model.to(device)
        self.dataloaders  = dataloaders
        self.dp           = dp_mechanism
        self.device       = device
        self.local_epochs = local_epochs
        self.fedprox_mu   = fedprox_mu

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=local_epochs * 10
        )
        self.criterion = nn.CrossEntropyLoss()

        self.last_train_loss = 1.0
        self.last_val_acc    = 0.0
        self.dataset_size    = len(dataloaders["train"].dataset)
        self.classes         = dataloaders.get("classes", [])

        # For Membership Inference tracking
        self._train_confidences: List[float] = []

        logger.info(
            f"Client {client_id}: {self.dataset_size} train | "
            f"{dataloaders['num_classes']} classes: {self.classes}"
        )

    def set_model_weights(self, global_weights: Dict):
        """
        Load backbone weights from the global model.
        SKIP any key whose shape does not match the local model
        (i.e. the classification head which is client-specific).
        """
        local_state  = self.model.state_dict()
        update_state = copy.deepcopy(local_state)

        skipped = []
        for k, v in global_weights.items():
            if k not in local_state:
                skipped.append(k)
                continue
            if v.shape != local_state[k].shape:
                # Head size mismatch — keep local head, skip this key
                skipped.append(k)
                continue
            update_state[k] = copy.deepcopy(v)

        self.model.load_state_dict(update_state)
        if skipped:
            logger.debug(f"{self.client_id}: kept local weights for {skipped}")

    def get_model_weights(self) -> Dict:
        return copy.deepcopy(self.model.state_dict())

    def train_local(self) -> Dict:
        """Local training with DP + optional FedProx proximal term."""
        initial_weights = copy.deepcopy(self.model.state_dict())
        global_params   = copy.deepcopy(list(self.model.parameters()))
        self.model.train()
        total_loss, total_n = 0.0, 0
        self._train_confidences = []

        for _ in range(self.local_epochs):
            for images, labels in self.dataloaders["train"]:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                # FedProx proximal term
                if self.fedprox_mu > 0.0:
                    prox = sum(
                        ((p - g.to(self.device)) ** 2).sum()
                        for p, g in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (self.fedprox_mu / 2.0) * prox

                loss.backward()
                self.dp.clip_gradients(self.model)
                self.dp.add_noise_to_gradients(self.model)
                self.optimizer.step()
                total_loss += loss.item() * images.size(0)
                total_n    += images.size(0)

                # Collect train confidences for MIA analysis
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)
                    self._train_confidences.extend(
                        probs.max(dim=1).values.cpu().tolist()
                    )

            self.scheduler.step()

        avg_loss = total_loss / (total_n + 1e-8)
        val_acc  = self.evaluate("val")
        self.last_train_loss = avg_loss
        self.last_val_acc    = val_acc

        cur   = self.model.state_dict()
        delta = {k: cur[k].float() - initial_weights[k].float() for k in cur}

        logger.info(
            f"  {self.client_id} — loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}"
        )
        return {
            "weight_delta": delta,
            "train_loss":   avg_loss,
            "val_accuracy": val_acc,
            "data_size":    self.dataset_size,
        }

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> float:
        self.model.eval()
        correct, total = 0, 0
        for images, labels in self.dataloaders[split]:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds  = self.model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        self.model.train()
        return correct / (total + 1e-8)

    @torch.no_grad()
    def test(self) -> Dict:
        """
        Full test evaluation collecting all data needed for every metric:
        y_true, y_pred, y_prob, inference time, confidences for MIA.
        """
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        test_confidences = []
        total_time, total_images = 0.0, 0

        for images, labels in self.dataloaders["test"]:
            images = images.to(self.device)
            labels = labels.to(self.device)

            t0     = time.perf_counter()
            logits = self.model(images)
            total_time   += time.perf_counter() - t0
            total_images += images.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            test_confidences.extend(
                probs.max(dim=1).values.cpu().tolist()
            )

        inf_ms = (total_time / max(total_images, 1)) * 1000

        return {
            "y_true":            all_labels,
            "y_pred":            all_preds,
            "y_prob":            all_probs,
            "accuracy":          sum(p == l for p, l in zip(all_preds, all_labels))
                                 / max(len(all_labels), 1),
            "classes":           self.classes,
            "total_samples":     len(all_labels),
            "inference_time_ms": round(inf_ms, 3),
            "train_confidences": self._train_confidences,
            "test_confidences":  test_confidences,
        }

    def get_rl_state_features(self) -> Dict:
        return {
            "loss":      self.last_train_loss,
            "accuracy":  self.last_val_acc,
            "data_size": self.dataset_size,
        }