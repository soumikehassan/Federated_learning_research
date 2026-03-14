"""
utils/metrics.py  ─ v2
Plotting utilities for all metric categories from research doc.
Generates publication-quality figures saved as PNG.
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed — pip install matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# Training curves (original 6-panel + method comparison)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history_path: str, save_dir: str):
    if not HAS_MPL:
        return
    with open(history_path) as f:
        history = json.load(f)

    rounds     = [r["round"]          for r in history]
    g_accs     = [r["global_acc"]     for r in history]
    l_accs     = [r["avg_local_acc"]  for r in history]
    losses     = [r["avg_local_loss"] for r in history]
    rewards    = [r["rl_reward"]      for r in history]
    epsilons   = [r["privacy_epsilon"]for r in history]
    method     = history[0].get("method", "FedAvg") if history else "FedAvg"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Dynamic Privacy-Preserving FL + RL  [{method}]\n"
        "Swin Transformer — Training Progress",
        fontsize=13, fontweight="bold"
    )

    def p(ax, y, title, ylabel, color):
        ax.plot(rounds, y, color=color, marker="o", linewidth=2, markersize=4)
        ax.set_title(title); ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    p(axes[0,0], g_accs,   "Global Accuracy",      "Accuracy", "#2196F3")
    p(axes[0,1], l_accs,   "Avg Local Accuracy",   "Accuracy", "#4CAF50")
    p(axes[0,2], losses,   "Avg Local Loss",        "Loss",     "#F44336")
    p(axes[1,0], rewards,  "RL Reward",             "Reward",   "#9C27B0")
    p(axes[1,1], epsilons, "Privacy Budget (eps)",  "eps",      "#FF9800")

    counts = {}
    for r in history:
        for c in r["selected_clients"]:
            counts[c] = counts.get(c, 0) + 1
    if counts:
        axes[1,2].bar(
            [str(c) for c in sorted(counts)],
            [counts[c] for c in sorted(counts)],
            color=["#2196F3","#4CAF50","#F44336"]
        )
        axes[1,2].set_title("Client Selection Frequency")
        axes[1,2].set_ylabel("Times Selected")
        axes[1,2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics bar chart (per client)
# ─────────────────────────────────────────────────────────────────────────────
def plot_classification_metrics(client_metrics: dict, save_dir: str, method="FedAvg"):
    if not HAS_MPL:
        return
    metric_names = ["accuracy","precision","recall","f1_score","specificity","cohen_kappa"]
    clients      = list(client_metrics.keys())
    x            = np.arange(len(metric_names))
    width        = 0.25
    colors       = ["#2196F3","#4CAF50","#F44336"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (cid, mets) in enumerate(client_metrics.items()):
        vals = [mets.get(m) or 0 for m in metric_names]
        bars = ax.bar(x + i*width, vals, width, label=cid, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Classification Metrics per Client  [{method}]")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(save_dir, f"classification_metrics_{method}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(client_metrics: dict, save_dir: str, method="FedAvg"):
    if not HAS_MPL:
        return
    clients = list(client_metrics.keys())
    fig, axes = plt.subplots(1, len(clients), figsize=(6*len(clients), 5))
    if len(clients) == 1:
        axes = [axes]

    for ax, (cid, mets) in zip(axes, client_metrics.items()):
        cm      = mets.get("confusion_matrix")
        classes = mets.get("classes", [])
        if cm is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title(cid)
            continue
        cm = np.array(cm)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{cid}\n{method}", fontsize=10)
        tick_marks = np.arange(len(cm))
        if classes:
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"Confusion Matrices  [{method}]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, f"confusion_matrices_{method}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Privacy-accuracy tradeoff (noise sensitivity: sigma = 0.5, 1.0, 2.0)
# ─────────────────────────────────────────────────────────────────────────────
def plot_privacy_accuracy_tradeoff(noise_results: dict, save_dir: str):
    """
    noise_results = {0.5: {"epsilon": x, "accuracy": y},
                     1.0: {"epsilon": x, "accuracy": y},
                     2.0: {"epsilon": x, "accuracy": y}}
    """
    if not HAS_MPL or not noise_results:
        return
    sigmas   = sorted(noise_results.keys())
    epsilons = [noise_results[s]["final_epsilon"] for s in sigmas]
    accs     = [noise_results[s]["final_accuracy"] for s in sigmas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(sigmas, accs, "o-", color="#2196F3", linewidth=2, markersize=8)
    axes[0].set_xlabel("Noise Multiplier (sigma)")
    axes[0].set_ylabel("Final Global Accuracy")
    axes[0].set_title("Noise Sensitivity: Accuracy vs Sigma")
    axes[0].grid(True, alpha=0.3)
    for s, a in zip(sigmas, accs):
        axes[0].annotate(f"{a:.3f}", (s, a), textcoords="offset points",
                         xytext=(0, 8), ha="center")

    axes[1].plot(epsilons, accs, "s-", color="#F44336", linewidth=2, markersize=8)
    axes[1].set_xlabel("Privacy Budget (epsilon)")
    axes[1].set_ylabel("Final Global Accuracy")
    axes[1].set_title("Privacy-Accuracy Tradeoff Curve")
    axes[1].axvline(x=10, color="grey", linestyle="--", alpha=0.5, label="eps=10 threshold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    for e, a in zip(epsilons, accs):
        axes[1].annotate(f"sigma={sigmas[epsilons.index(e)]}", (e, a),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.suptitle("Differential Privacy Noise Sensitivity Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "privacy_accuracy_tradeoff.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation method comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_method_comparison(comparison_data: dict, save_dir: str):
    """
    comparison_data = {
      "FedAvg":    {"global_acc": 0.72, "client_0": 0.80, "client_1": 0.55, "client_2": 0.88},
      "FedProx":   {...},
      "FedMedian": {...},
      "SCAFFOLD":  {...},
    }
    """
    if not HAS_MPL or not comparison_data:
        return
    methods  = list(comparison_data.keys())
    metrics  = ["global_acc", "client_0", "client_1", "client_2"]
    labels   = ["Global Acc", "Alzheimer", "Retinal", "TB"]
    x        = np.arange(len(methods))
    width    = 0.2
    colors   = ["#2196F3","#4CAF50","#F44336","#FF9800"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [comparison_data[m].get(metric, 0) for m in methods]
        bars = ax.bar(x + i*width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Aggregation Method Comparison: FedAvg vs FedProx vs FedMedian vs SCAFFOLD")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(save_dir, "method_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fairness radar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_fairness(client_accuracies: dict, methods_fairness: dict, save_dir: str):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: per-client accuracy bar chart
    cids   = list(client_accuracies.keys())
    accs   = list(client_accuracies.values())
    colors = ["#2196F3","#4CAF50","#F44336"]
    bars   = axes[0].bar(cids, accs, color=colors[:len(cids)], alpha=0.85)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{acc:.3f}", ha="center", va="bottom")
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=np.mean(accs), color="black", linestyle="--",
                    alpha=0.5, label=f"Mean={np.mean(accs):.3f}")
    axes[0].set_title("Per-Client Test Accuracy (Fairness)"); axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: fairness gap per method
    if methods_fairness:
        mnames = list(methods_fairness.keys())
        gaps   = [methods_fairness[m].get("fairness_gap", 0) for m in mnames]
        axes[1].bar(mnames, gaps, color="#9C27B0", alpha=0.8)
        axes[1].set_title("Fairness Gap per Aggregation Method\n(lower = fairer)")
        axes[1].set_ylabel("Fairness Gap (Best Acc - Worst Acc)")
        axes[1].axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="Threshold=0.15")
        axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Fairness Analysis Across Clients and Methods", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "fairness_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Final text report
# ─────────────────────────────────────────────────────────────────────────────
def print_final_report(history, client_test_results, selection_stats,
                       privacy_report, method="FedAvg", client_ids=None):
    print("\n" + "="*65)
    print(f"  FINAL REPORT  [{method}]  FL + RL + DP + Swin Transformer")
    print("="*65)
    if history:
        accs = [r["global_acc"] for r in history]
        print(f"\n  Global Acc — Best: {max(accs):.4f}  Final: {accs[-1]:.4f}")

    print("\n  Client Test Results:")
    for cid, res in client_test_results.items():
        print(f"    {cid}: acc={res.get('accuracy',0):.4f} | "
              f"f1={res.get('f1_score') or 0:.4f} | "
              f"auc={res.get('auc_roc') or 'N/A'}")

    if selection_stats:
        print("\n  RL Client Selection:")
        n = len(selection_stats)
        for i in range(n):
            key = f"client_{i}"
            if key not in selection_stats:
                continue
            s = selection_stats[key]
            label = client_ids[i] if client_ids and i < len(client_ids) else key
            print(f"    {label}: {s['times_selected']}x ({s['selection_rate']*100:.1f}%)")

    eps = privacy_report.get("epsilon", "N/A")
    dlt = privacy_report.get("delta", "N/A")
    eps_str = f"{eps:.4f}" if isinstance(eps, float) else str(eps)
    print(f"\n  Privacy: eps={eps_str}, delta={dlt}")
    print("="*65 + "\n")
