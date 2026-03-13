"""
utils/evaluation_metrics.py  ─ v2
Complete evaluation metrics covering ALL categories from the research document:

  1. Classification:  Accuracy, Precision, Recall, F1, AUC-ROC,
                      Specificity, Cohen's Kappa, Confusion Matrix
  2. Privacy:         epsilon per round, Privacy-Accuracy tradeoff,
                      Noise Multiplier Sensitivity, Membership Inference
  3. RL:              Cumulative reward, Selection frequency, Diversity entropy,
                      Convergence speed, RL vs Random, RL vs Full Participation
  4. FL:              Global convergence rate, Communication cost, Client drift,
                      Local vs Global accuracy gap, Rounds to convergence
  5. Fairness:        Per-client variance, Worst-client acc, Fairness gap,
                      Participation equity
  6. Computational:   Time per round, Total time, Model params, Memory
  7. Ablation:        No-RL, No-DP, No-Pretrain, Different noise, Fewer rounds
  8. Baseline:        Centralized, Local-only, FedAvg-all, Random, FedProx

All results saved to CSV automatically.
"""

import os
import csv
import time
import json
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, cohen_kappa_score,
        confusion_matrix, classification_report
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not installed — pip install scikit-learn")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_classification_metrics(
    y_true, y_pred, y_prob=None, classes=None
) -> Dict:
    """
    Full classification suite for one client.
    y_true / y_pred : lists or arrays of integer labels
    y_prob          : (N, C) probability array — needed for AUC-ROC
    classes         : list of class name strings
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_cls  = len(np.unique(y_true))
    m      = {}

    # ── Accuracy ──────────────────────────────────────────────────────────────
    m["accuracy"] = float(np.mean(y_true == y_pred))

    if not HAS_SKLEARN:
        for k in ("precision","recall","f1_score","specificity",
                  "cohen_kappa","auc_roc","confusion_matrix"):
            m[k] = None
        return m

    avg = "macro"
    m["precision"]   = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    m["recall"]      = float(recall_score   (y_true, y_pred, average=avg, zero_division=0))
    m["f1_score"]    = float(f1_score       (y_true, y_pred, average=avg, zero_division=0))
    m["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # ── Specificity (macro) ───────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(len(cm)):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp + 1e-8))
    m["specificity"] = float(np.mean(specs))

    # ── AUC-ROC ──────────────────────────────────────────────────────────────
    m["auc_roc"] = None
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            if n_cls == 2:
                m["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                m["auc_roc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except Exception as e:
            logger.warning(f"AUC-ROC skipped: {e}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    m["confusion_matrix"] = cm.tolist()

    # ── Per-class breakdown ───────────────────────────────────────────────────
    if classes:
        report = classification_report(
            y_true, y_pred, target_names=classes,
            output_dict=True, zero_division=0
        )
        m["per_class"] = {
            c: {"precision": report[c]["precision"],
                "recall":    report[c]["recall"],
                "f1":        report[c]["f1-score"],
                "support":   report[c]["support"]}
            for c in classes if c in report
        }

    return m


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PRIVACY METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_privacy_metrics(epsilon_per_round: List[float],
                             accuracy_per_round: List[float]) -> Dict:
    """Epsilon tracking + privacy-accuracy tradeoff curve."""
    return {
        "epsilon_per_round":      epsilon_per_round,
        "final_epsilon":          round(epsilon_per_round[-1], 4)  if epsilon_per_round else None,
        "max_epsilon":            round(max(epsilon_per_round), 4) if epsilon_per_round else None,
        "epsilon_below_10":       all(e < 10 for e in epsilon_per_round),
        "accuracy_per_round":     accuracy_per_round,
        "privacy_accuracy_curve": list(zip(epsilon_per_round, accuracy_per_round)),
    }


def compute_membership_inference_resistance(
    model_confidence_train: List[float],
    model_confidence_test:  List[float]
) -> Dict:
    """
    Estimate Membership Inference Attack (MIA) resistance.
    A well-private model should have similar confidence on train vs test.
    AUC of attacker close to 0.5 = strong privacy.
    """
    if not HAS_SKLEARN or not model_confidence_train or not model_confidence_test:
        return {"mia_auc": None, "mia_gap": None}
    try:
        from sklearn.metrics import roc_auc_score as ras
        labels = [1]*len(model_confidence_train) + [0]*len(model_confidence_test)
        scores = list(model_confidence_train) + list(model_confidence_test)
        mia_auc = float(ras(labels, scores))
    except Exception:
        mia_auc = None
    gap = (np.mean(model_confidence_train) - np.mean(model_confidence_test)
           if model_confidence_train and model_confidence_test else None)
    return {
        "mia_auc":  round(mia_auc, 4) if mia_auc else None,
        "mia_gap":  round(float(gap), 4) if gap is not None else None,
        "interpretation": "AUC ~0.5 = strong privacy; AUC ~1.0 = weak privacy",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RL METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_rl_metrics(reward_history:    List[float],
                       selection_history: List[List[int]],
                       accuracy_history:  List[float],
                       num_clients:       int) -> Dict:
    rewards = np.array(reward_history)

    # Cumulative rewards
    cumulative = np.cumsum(rewards).tolist()

    # Client selection frequency + diversity entropy
    counts = {i: 0 for i in range(num_clients)}
    for sel in selection_history:
        for c in sel:
            counts[c] += 1
    total = max(len(selection_history), 1)
    rates = {f"client_{i}": round(counts[i] / total, 4) for i in range(num_clients)}
    probs = np.array([counts[i] / total for i in range(num_clients)])
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

    # Convergence speed — round where accuracy first hits 80% of peak
    target      = max(accuracy_history) * 0.8 if accuracy_history else 0
    conv_round  = next((i+1 for i, a in enumerate(accuracy_history) if a >= target), None)

    return {
        "reward_per_round":     [round(r, 4) for r in reward_history],
        "cumulative_rewards":   [round(r, 4) for r in cumulative],
        "total_reward":         round(float(rewards.sum()), 4),
        "mean_reward":          round(float(rewards.mean()), 4),
        "selection_rates":      rates,
        "selection_entropy":    round(entropy, 4),
        "convergence_round":    conv_round,
        "note_rl_vs_random":    "Compare total_reward and convergence_round vs RandomSelector",
        "note_rl_vs_full":      "Compare convergence_round vs FedAvg with all clients",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FL METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_fl_metrics(global_acc_history:       List[float],
                       local_acc_history:         List[float],
                       selected_clients_history:  List[List],
                       model_size_mb:             float,
                       time_per_round:            List[float]) -> Dict:
    g = np.array(global_acc_history)
    l = np.array(local_acc_history)

    # Convergence rate: rounds to reach 80% of best
    best   = g.max()
    target = best * 0.8
    conv80 = next((i+1 for i, a in enumerate(global_acc_history) if a >= target), None)

    # Stable convergence: delta < 0.005 for 3 consecutive rounds
    stable = None
    for i in range(2, len(g)):
        if abs(g[i]-g[i-1]) < 0.005 and abs(g[i-1]-g[i-2]) < 0.005:
            stable = i + 1
            break

    # Communication cost
    avg_n  = np.mean([len(s) for s in selected_clients_history])
    comm   = round(model_size_mb * len(global_acc_history) * avg_n, 2)

    return {
        "convergence_round_80pct":  conv80,
        "stable_convergence_round": stable,
        "communication_cost_mb":    comm,
        "avg_clients_per_round":    round(float(avg_n), 2),
        "local_vs_global_gap":      [round(float(x), 4) for x in (l - g).tolist()],
        "client_drift_proxy":       round(float(np.std(l)), 4),
        "best_global_acc":          round(float(best), 4),
        "final_global_acc":         round(float(g[-1]), 4),
        "total_rounds":             len(global_acc_history),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FAIRNESS METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_fairness_metrics(client_accuracies: Dict[str, float],
                             selection_rates:   Dict[str, float]) -> Dict:
    accs  = list(client_accuracies.values())
    rates = list(selection_rates.values())
    return {
        "per_client_accuracy":      {k: round(v, 4) for k, v in client_accuracies.items()},
        "accuracy_mean":            round(float(np.mean(accs)), 4),
        "accuracy_std":             round(float(np.std(accs)),  4),
        "worst_client_accuracy":    round(float(min(accs)), 4),
        "best_client_accuracy":     round(float(max(accs)), 4),
        "fairness_gap":             round(float(max(accs) - min(accs)), 4),
        "participation_equity_std": round(float(np.std(rates)), 4),
        "fairness_pass":            (max(accs) - min(accs)) < 0.15,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  COMPUTATIONAL METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_computational_metrics(time_per_round: List[float],
                                  model_params:   int,
                                  inference_times: Optional[List[float]] = None) -> Dict:
    try:
        import psutil
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        mem_mb = None

    return {
        "time_per_round_sec":    [round(t, 2) for t in time_per_round],
        "avg_time_per_round_sec": round(float(np.mean(time_per_round)), 2),
        "total_training_time_sec":round(float(np.sum(time_per_round)), 2),
        "model_params_millions":  round(model_params / 1e6, 2),
        "peak_memory_mb":         round(mem_mb, 1) if mem_mb else None,
        "inference_times_ms":     inference_times,
        "avg_inference_ms":       round(float(np.mean(inference_times)), 3) if inference_times else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV RESULTS LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class CSVResultsLogger:
    """
    Saves every metric category to its own CSV file automatically.
    Call log_round() after each round, and log_final() at the end.
    """

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self._round_rows  = []
        self._client_rows = []

    # ── Per-round log ──────────────────────────────────────────────────────────
    def log_round(self, round_num: int, method: str, selected_clients: List,
                  client_updates: List, global_acc: float, rl_reward: float,
                  rl_loss: float, privacy_eps: float, time_sec: float):
        avg_loss = np.mean([u["train_loss"]  for u in client_updates])
        avg_acc  = np.mean([u["val_accuracy"]for u in client_updates])
        row = {
            "round":              round_num,
            "method":             method,
            "selected_clients":   str(selected_clients),
            "num_selected":       len(selected_clients),
            "avg_local_loss":     round(avg_loss,    4),
            "avg_local_accuracy": round(avg_acc,     4),
            "global_accuracy":    round(global_acc,  4),
            "rl_reward":          round(rl_reward,   4),
            "rl_loss":            round(rl_loss,     6),
            "privacy_epsilon":    round(privacy_eps, 4),
            "time_seconds":       round(time_sec,    2),
        }
        self._round_rows.append(row)
        self._write_csv("round_metrics.csv", self._round_rows)

    # ── Per-client final test ──────────────────────────────────────────────────
    def log_client_test(self, client_id: str, method: str, metrics: Dict):
        row = {
            "client_id":    client_id,
            "method":       method,
            "accuracy":     round(metrics.get("accuracy")    or 0, 4),
            "precision":    round(metrics.get("precision")   or 0, 4),
            "recall":       round(metrics.get("recall")      or 0, 4),
            "f1_score":     round(metrics.get("f1_score")    or 0, 4),
            "specificity":  round(metrics.get("specificity") or 0, 4),
            "cohen_kappa":  round(metrics.get("cohen_kappa") or 0, 4),
            "auc_roc":      round(metrics.get("auc_roc")     or 0, 4),
        }
        self._client_rows.append(row)
        self._write_csv("client_test_metrics.csv", self._client_rows)

    # ── Privacy ────────────────────────────────────────────────────────────────
    def log_privacy(self, method: str, noise_mult: float, privacy: Dict):
        row = {
            "method":        method,
            "noise_mult":    noise_mult,
            "final_epsilon": privacy.get("final_epsilon"),
            "max_epsilon":   privacy.get("max_epsilon"),
            "epsilon_ok":    privacy.get("epsilon_below_10"),
        }
        self._append_csv("privacy_metrics.csv", row)

    # ── Fairness ───────────────────────────────────────────────────────────────
    def log_fairness(self, method: str, fairness: Dict):
        row = {"method": method}
        row.update({k: v for k, v in fairness.items()
                    if not isinstance(v, dict)})
        self._append_csv("fairness_metrics.csv", row)

    # ── RL ─────────────────────────────────────────────────────────────────────
    def log_rl(self, rl: Dict):
        row = {
            "total_reward":       rl.get("total_reward"),
            "mean_reward":        rl.get("mean_reward"),
            "selection_entropy":  rl.get("selection_entropy"),
            "convergence_round":  rl.get("convergence_round"),
        }
        row.update(rl.get("selection_rates", {}))
        self._append_csv("rl_metrics.csv", row)

    # ── Computational ──────────────────────────────────────────────────────────
    def log_computational(self, comp: Dict):
        row = {
            "avg_time_per_round_sec":  comp.get("avg_time_per_round_sec"),
            "total_training_time_sec": comp.get("total_training_time_sec"),
            "model_params_millions":   comp.get("model_params_millions"),
            "peak_memory_mb":          comp.get("peak_memory_mb"),
            "avg_inference_ms":        comp.get("avg_inference_ms"),
        }
        self._append_csv("computational_metrics.csv", row)

    # ── FL metrics ─────────────────────────────────────────────────────────────
    def log_fl(self, method: str, fl: Dict):
        row = {"method": method}
        row.update({k: v for k, v in fl.items()
                    if not isinstance(v, list)})
        self._append_csv("fl_metrics.csv", row)

    # ── Method comparison table (paper table) ─────────────────────────────────
    def log_method_comparison(self, method: str, client_acc: Dict[str, float],
                              global_acc: float, privacy_eps: float,
                              fairness_gap: float, conv_round: int):
        row = {
            "method":        method,
            "alzheimer_acc": round(client_acc.get("client_0", 0), 4),
            "retinal_acc":   round(client_acc.get("client_1", 0), 4),
            "tb_acc":        round(client_acc.get("client_2", 0), 4),
            "global_acc":    round(global_acc,    4),
            "privacy_eps":   round(privacy_eps,   4),
            "fairness_gap":  round(fairness_gap,  4),
            "conv_round":    conv_round,
        }
        self._append_csv("method_comparison.csv", row)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _write_csv(self, filename: str, rows: List[Dict]):
        if not rows:
            return
        path = os.path.join(self.results_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    def _append_csv(self, filename: str, row: Dict):
        path   = os.path.join(self.results_dir, filename)
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not exists:
                w.writeheader()
            w.writerow(row)

    def print_summary(self):
        print("\n  [CSV] Results saved:")
        for fname in sorted(os.listdir(self.results_dir)):
            if fname.endswith(".csv"):
                kb = os.path.getsize(os.path.join(self.results_dir, fname)) / 1024
                print(f"    {fname:<40} ({kb:.1f} KB)")
