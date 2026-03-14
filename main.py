"""
main.py  ─ v2
Upgraded full federated learning training loop.

New in v2:
  - Runs ALL aggregation methods (FedAvg, FedProx, FedMedian, SCAFFOLD)
  - SMPC / Secure Aggregation support
  - Full evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC,
    Specificity, Cohen's Kappa, Confusion Matrix, Privacy, RL, FL, Fairness,
    Computational metrics)
  - All results automatically saved to CSV
  - Noise sensitivity experiment (sigma = 0.5, 1.0, 2.0)
  - All comparison plots generated

Usage:
    python main.py                                          # all methods
    python main.py --method FedAvg                         # single method
    python main.py --rounds 5 --max-samples 50             # quick test
    python main.py --no-dp                                  # disable DP
    python main.py --use-smpc                               # secure aggregation
    python main.py --noise-sweep                            # sigma 0.5/1.0/2.0
"""

import argparse
import logging
import os
import sys
import time
import copy
import torch
import random
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import config
from data.dataset                 import get_client_dataloaders, get_client_dataloaders_with_partition
from models.swin_transformer      import build_swin_model
from clients.federated_client     import FederatedClient
from server.federated_server      import FederatedServer
from server.aggregation_methods   import get_aggregator
from utils.differential_privacy   import DPMechanism
from utils.secure_aggregation     import SecureAggregator
from utils.rl_client_selector     import RLClientSelector
from utils.evaluation_metrics     import (
    compute_classification_metrics,
    compute_privacy_metrics,
    compute_rl_metrics,
    compute_fl_metrics,
    compute_fairness_metrics,
    compute_computational_metrics,
    compute_membership_inference_resistance,
    CSVResultsLogger,
)
from utils.metrics import (
    plot_training_curves, plot_classification_metrics,
    plot_confusion_matrices, plot_privacy_accuracy_tradeoff,
    plot_method_comparison, plot_fairness, print_final_report,
)


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu"); print("  Device: CPU")
    return d

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8")],
    )

def make_dummy_loaders(cid, num_classes, batch_size, image_size):
    from torch.utils.data import TensorDataset, DataLoader
    n = 200
    x = torch.randn(n, 3, image_size, image_size)
    y = torch.randint(0, num_classes, (n,))
    return {
        "train":       DataLoader(TensorDataset(x[:160], y[:160]), batch_size=batch_size, shuffle=True),
        "val":         DataLoader(TensorDataset(x[160:180], y[160:180]), batch_size=batch_size),
        "test":        DataLoader(TensorDataset(x[180:], y[180:]), batch_size=batch_size),
        "num_classes": num_classes,
        "classes":     [f"class_{i}" for i in range(num_classes)],
        "dataset_size":160,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE METHOD TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────────
def run_one_method(args, method: str, loaders, models_init, device,
                   csv_logger: CSVResultsLogger,
                   noise_mult: float = None) -> dict:
    """
    Train one full federated round loop for one aggregation method.
    Returns final metrics dict.
    """
    sigma = noise_mult if noise_mult is not None else (args.sigma if hasattr(args, 'sigma') and args.sigma is not None else config.DP_NOISE_MULT)
    print(f"\n{'='*65}")
    print(f"  Method: {method}  |  sigma={sigma}  |  DP={'off' if args.no_dp else 'on'}")
    print(f"{'='*65}")

    # ── DP setup ──────────────────────────────────────────────────────────────
    dp_map = {
        cid: DPMechanism(sigma, config.DP_MAX_GRAD_NORM,
                         config.DP_DELTA, enabled=not args.no_dp)
        for cid in config.DATASET_PATHS
    }

    # ── SMPC setup ────────────────────────────────────────────────────────────
    smpc = None
    if args.use_smpc:
        smpc = SecureAggregator(
            num_clients=config.NUM_CLIENTS,
            num_shares=config.SMPC_NUM_SHARES,
            reconstruction_threshold=config.SMPC_RECONSTRUCTION_THRESHOLD,
            device=str(device),
        )

    # ── RL selector ───────────────────────────────────────────────────────────
    # If selection mode is not 'rl', RL will effectively act as a pass-through
    # for random or all selection, but we maintain the class for metrics.
    rl = RLClientSelector(
        num_clients=config.NUM_CLIENTS,
        hidden_dim=config.RL_HIDDEN_DIM, lr=config.RL_LR,
        gamma=config.RL_GAMMA,
        epsilon_start=config.RL_EPSILON_START if args.selection == 'rl' else 1.0,
        epsilon_end=config.RL_EPSILON_END if args.selection == 'rl' else 1.0,
        epsilon_decay=config.RL_EPSILON_DECAY,
        buffer_size=config.RL_BUFFER_SIZE,
        batch_size=config.RL_BATCH_SIZE,
        min_clients=config.RL_MIN_CLIENTS,
        device=str(device),
    )

    # ── Fresh clients (deep copy models so each method starts from same init) ─
    clients = {
        cid: FederatedClient(
            cid, copy.deepcopy(models_init[cid]), loaders[cid],
            dp_map[cid], device,
            lr=args.lr, weight_decay=config.WEIGHT_DECAY,
            local_epochs=args.local_epochs,
            fedprox_mu=(config.FEDPROX_MU if method == "FedProx" else 0.0),
        )
        for cid in config.DATASET_PATHS
    }

    # ── Server ────────────────────────────────────────────────────────────────
    method_results_dir = os.path.join(args.output_dir, method)
    os.makedirs(method_results_dir, exist_ok=True)
    first_cid = next(iter(config.DATASET_PATHS))

    server = FederatedServer(
        copy.deepcopy(models_init[first_cid]),
        dp_map[first_cid], device,
        results_dir=method_results_dir,
        aggregation_method=method,
        fedprox_mu=config.FEDPROX_MU,
        use_smpc=args.use_smpc,
        smpc_aggregator=smpc,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    client_ids = list(config.DATASET_PATHS.keys())  # ordered: client_0_0, client_0_1, ...
    prev_acc, prev_st, prev_sel = 0.0, None, None
    epsilon_history, acc_history = [], []
    time_per_round, selection_history, reward_history = [], [], []

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        print(f"  [{method}] Round {rnd}/{args.rounds}")

        stats = {cid: c.get_rl_state_features() for cid, c in clients.items()}
        # RL expects keys "client_0".."client_{N-1}"; map actual ids to index
        stats_for_rl = {f"client_{i}": stats[cid] for i, cid in enumerate(client_ids)}
        state = rl.build_state(stats_for_rl)

        # Determine selection indices (0..NUM_CLIENTS-1), then map to actual client ids
        if args.selection == 'all':
            sel_idx = list(range(config.NUM_CLIENTS))
        elif args.selection == 'random':
            sel_idx = random.sample(range(config.NUM_CLIENTS), config.RL_MIN_CLIENTS)
        else:
            sel_idx = rl.select_clients(stats_for_rl, rnd)

        sel_cids = [client_ids[i] for i in sel_idx]
        selection_history.append(sel_idx)
        print(f"    Selected: {sel_cids} (Mode={args.selection}, eps={rl.epsilon:.3f})")

        updates = []
        for cid in sel_cids:
            upd = clients[cid].train_local()
            updates.append(upd)

        # Aggregate
        new_weights = server.aggregate(updates, selected_cids=sel_cids)
        for cid in clients:
            clients[cid].set_model_weights(new_weights)

        global_acc = float(np.average(
            [u["val_accuracy"] for u in updates],
            weights=[u["data_size"] for u in updates]
        ))
        # Get per-client accuracies for fairness reward (use integer indices)
        client_accs = {idx: u["val_accuracy"] for idx, u in zip(sel_idx, updates)}
        # Map client IDs to domain IDs for RL
        cid_to_idx = {cid: i for i, cid in enumerate(config.DATASET_PATHS.keys())}
        client_domains = {cid_to_idx[cid]: dom for cid, dom in config.CLIENT_DOMAINS.items()}
        reward  = rl.compute_reward(prev_acc, global_acc, sel_idx, rnd,
                                    client_accs=client_accs,
                                    client_domains=client_domains)
        nxt_st  = rl.build_state({f"client_{i}": clients[cid].get_rl_state_features() for i, cid in enumerate(client_ids)})
        if prev_st is not None:
            rl.store_transition(prev_st, prev_sel, reward, state, False)
        rl_loss = rl.update()
        prev_acc, prev_st, prev_sel = global_acc, state, sel_idx

        sr  = args.batch_size / max(sum(clients[c].dataset_size for c in sel_cids), 1)
        pr  = dp_map[first_cid].get_privacy_report(rnd * args.local_epochs, sr)
        eps = pr["epsilon"]

        t_elapsed = time.time() - t0
        time_per_round.append(t_elapsed)
        epsilon_history.append(eps)
        acc_history.append(global_acc)
        reward_history.append(reward)

        server.log_round(rnd, sel_cids, updates, global_acc, reward, rl_loss, eps)
        csv_logger.log_round(rnd, method, sel_cids, updates,
                             global_acc, reward, rl_loss, eps, t_elapsed)
        print(f"    GlobalAcc={global_acc:.4f} | Reward={reward:.4f} | "
              f"eps={eps:.4f} | {t_elapsed:.1f}s")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n  [{method}] Final test evaluation...")
    raw_test    = {}   # y_true, y_pred, y_prob, etc.
    full_metrics= {}   # computed sklearn metrics

    for cid, client in clients.items():
        raw = client.test()
        raw_test[cid] = raw
        mets = compute_classification_metrics(
            raw["y_true"], raw["y_pred"], raw["y_prob"],
            classes=raw["classes"]
        )
        mets["classes"] = raw["classes"]
        full_metrics[cid] = mets
        csv_logger.log_client_test(cid, method, mets)
        print(f"    {cid}: acc={mets['accuracy']:.4f} | "
              f"f1={mets.get('f1_score') or 0:.4f} | "
              f"auc={mets.get('auc_roc') or 'N/A'}")

    # ── Save history + plots ──────────────────────────────────────────────────
    hist_path = server.save_history()
    plot_training_curves(hist_path, method_results_dir)
    plot_classification_metrics(full_metrics, method_results_dir, method)
    plot_confusion_matrices(full_metrics, method_results_dir, method)

    # ── Privacy metrics ───────────────────────────────────────────────────────
    priv = compute_privacy_metrics(epsilon_history, acc_history)
    csv_logger.log_privacy(method, sigma, priv)

    # ── MIA resistance ────────────────────────────────────────────────────────
    for cid in clients:
        mia = compute_membership_inference_resistance(
            raw_test[cid].get("train_confidences", []),
            raw_test[cid].get("test_confidences",  []),
        )
        print(f"    {cid} MIA AUC={mia.get('mia_auc','N/A')} "
              f"(closer to 0.5 = stronger privacy)")

    # ── RL metrics ────────────────────────────────────────────────────────────
    rl_mets = compute_rl_metrics(
        reward_history, selection_history, acc_history, config.NUM_CLIENTS
    )
    csv_logger.log_rl(rl_mets)

    # ── FL metrics ────────────────────────────────────────────────────────────
    local_acc_hist = [r["avg_local_acc"] for r in server.round_history]
    model_mb = sum(p.numel() * 4 for p in
                   list(clients[first_cid].model.parameters())) / 1e6
    fl_mets = compute_fl_metrics(
        acc_history, local_acc_hist, selection_history,
        model_mb, time_per_round
    )
    csv_logger.log_fl(method, fl_mets)

    # ── Fairness metrics ──────────────────────────────────────────────────────
    client_accs  = {cid: full_metrics[cid]["accuracy"] for cid in full_metrics}
    fair = compute_fairness_metrics(client_accs, rl_mets["selection_rates"])
    csv_logger.log_fairness(method, fair)

    # ── Computational metrics ─────────────────────────────────────────────────
    model_params = sum(p.numel() for p in clients[first_cid].model.parameters())
    inf_times    = [raw_test[cid]["inference_time_ms"] for cid in raw_test]
    comp = compute_computational_metrics(time_per_round, model_params, inf_times)
    csv_logger.log_computational(comp)

    # ── Method comparison table entry ─────────────────────────────────────────
    csv_logger.log_method_comparison(
        method, client_accs,
        global_acc   = acc_history[-1],
        privacy_eps  = epsilon_history[-1],
        fairness_gap = fair["fairness_gap"],
        conv_round   = fl_mets["convergence_round_80pct"] or args.rounds,
    )

    # ── RL save ───────────────────────────────────────────────────────────────
    rl.save(os.path.join(method_results_dir, "rl_agent.pt"))

    final_pr = dp_map[first_cid].get_privacy_report(
        args.rounds * args.local_epochs,
        args.batch_size / max(clients[first_cid].dataset_size, 1)
    )
    print_final_report(server.round_history, full_metrics,
                       rl.get_selection_stats(), final_pr, method,
                       client_ids=client_ids)

    return {
        "method":          method,
        "full_metrics":    full_metrics,
        "privacy":         priv,
        "fairness":        fair,
        "fl":              fl_mets,
        "rl":              rl_mets,
        "global_acc":      acc_history[-1],
        "noise_mult":      sigma,
        "epsilon_history": epsilon_history,
        "acc_history":     acc_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    print("\n" + "="*65)
    print("  Dynamic Privacy-Preserving Federated Learning  v2")
    print("  Swin Transformer + RL + DP/SMPC + Multi-Method Comparison")
    print("="*65)

    set_seed(config.SEED)
    device = get_device()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    setup_logging(os.path.join(config.RESULTS_DIR, "training.log"))

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading datasets with Dirichlet partitioning...")
    loaders = {}
    for cid, path in config.DATASET_PATHS.items():
        alpha = config.DOMAIN_DIRICHLET_ALPHA.get(cid, 0.5)
        try:
            loaders[cid] = get_client_dataloaders_with_partition(
                cid, path,
                alpha=alpha,
                client_idx=int(cid.split("_")[-1]),  # 0 or 1 within domain
                num_domain_clients=2,
                image_size=config.IMAGE_SIZE, batch_size=args.batch_size,
                train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO,
                num_workers=config.NUM_WORKERS, seed=config.SEED,
                max_per_class=args.max_samples,
            )
            print(f"  {cid}: {loaders[cid]['dataset_size']} train | "
                  f"{loaders[cid]['num_classes']} classes | α={alpha}")
        except FileNotFoundError:
            nc = config.NUM_CLASSES_PER_CLIENT.get(cid, 2)
            print(f"  {cid}: path not found — using synthetic ({nc} classes)")
            loaders[cid] = make_dummy_loaders(cid, nc, args.batch_size, config.IMAGE_SIZE)

    # ── Models ────────────────────────────────────────────────────────────────
    print("\n[2/4] Building Swin Transformer models...")
    models_init = {}
    for cid in config.DATASET_PATHS:
        nc = loaders[cid]["num_classes"]
        m  = build_swin_model(nc, pretrained=not args.no_pretrain,
                              model_size=args.model_size)
        models_init[cid] = m
        p = sum(x.numel() for x in m.parameters()) / 1e6
        print(f"  {cid}: {nc} classes | {p:.1f}M params")

    # ── CSV Logger ────────────────────────────────────────────────────────────
    csv_logger = CSVResultsLogger(args.output_dir)

    # ── Determine which methods to run ────────────────────────────────────────
    if args.method == "all":
        methods = config.AGGREGATION_METHODS
    else:
        methods = [args.method]

    print(f"\n[3/4] Running {len(methods)} aggregation method(s): {methods}")

    # ── Main experiment loop ──────────────────────────────────────────────────
    all_results = {}
    for method in methods:
        result = run_one_method(
            args, method, loaders, models_init, device, csv_logger
        )
        all_results[method] = result

    # ── Noise sensitivity sweep ───────────────────────────────────────────────
    noise_results = {}
    if args.noise_sweep:
        print("\n[Noise Sweep] sigma = 0.5, 1.0, 2.0 ...")
        for sigma in config.DP_NOISE_VARIANTS:
            print(f"\n  Noise sweep: sigma={sigma}")
            res = run_one_method(
                args, "FedAvg", loaders, models_init, device,
                csv_logger, noise_mult=sigma
            )
            noise_results[sigma] = {
                "final_epsilon":  res["epsilon_history"][-1],
                "final_accuracy": res["acc_history"][-1],
            }
        plot_privacy_accuracy_tradeoff(noise_results, config.RESULTS_DIR)

    # ── Cross-method comparison plots ─────────────────────────────────────────
    print("\n[4/4] Generating comparison plots...")
    # One representative client per domain for comparison plot (client_0_0, client_1_0, client_2_0)
    rep = {"client_0": "client_0_0", "client_1": "client_1_0", "client_2": "client_2_0"}
    comp_data = {}
    for m, r in all_results.items():
        comp_data[m] = {"global_acc": r["global_acc"]}
        for plot_key, cid in rep.items():
            comp_data[m][plot_key] = r["full_metrics"].get(cid, {}).get("accuracy", 0.0)
    plot_method_comparison(comp_data, config.RESULTS_DIR)

    # Fairness comparison across methods
    methods_fairness = {m: r["fairness"] for m, r in all_results.items()}
    best_method      = max(all_results, key=lambda m: all_results[m]["global_acc"])
    plot_fairness(
        {cid: all_results[best_method]["full_metrics"][cid]["accuracy"]
         for cid in config.DATASET_PATHS},
        methods_fairness,
        config.RESULTS_DIR,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EXPERIMENT COMPLETE")
    print("="*65)
    print("\n  Method Comparison:")
    print(f"  {'Method':<12} {'GlobalAcc':>10} {'FairnessGap':>12} {'Epsilon':>9}")
    print(f"  {'-'*47}")
    for m, r in all_results.items():
        print(f"  {m:<12} {r['global_acc']:>10.4f} "
              f"{r['fairness']['fairness_gap']:>12.4f} "
              f"{r['privacy']['final_epsilon']:>9.4f}")

    csv_logger.print_summary()
    print(f"\n  All results saved to: {config.RESULTS_DIR}/")
    print("  Each method has its own subfolder with training_history.json,")
    print("  training_curves.png, confusion_matrices.png, best_model.pt")
    print("\n  Top-level CSV files for your paper tables:")
    print("    round_metrics.csv, client_test_metrics.csv,")
    print("    privacy_metrics.csv, fairness_metrics.csv,")
    print("    rl_metrics.csv, fl_metrics.csv,")
    print("    computational_metrics.csv, method_comparison.csv\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds",       type=int,   default=config.NUM_ROUNDS)
    p.add_argument("--local-epochs", type=int,   default=config.LOCAL_EPOCHS)
    p.add_argument("--batch-size",   type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=config.LEARNING_RATE)
    p.add_argument("--model-size",   type=str,   default="tiny",
                   choices=["tiny","small","base"])
    p.add_argument("--method",       type=str,   default="all",
                   help="FedAvg | FedProx | FedMedian | SCAFFOLD | all")
    p.add_argument("--no-dp",        action="store_true", help="Disable DP")
    p.add_argument("--use-smpc",     action="store_true", help="Enable SMPC aggregation")
    p.add_argument("--no-pretrain",  action="store_true", help="No ImageNet weights")
    p.add_argument("--max-samples",  type=int,   default=None)
    p.add_argument("--noise-sweep",  action="store_true",
                   help="Run noise sensitivity: sigma=0.5, 1.0, 2.0")
    p.add_argument("--seed",         type=int,   default=config.SEED, help="Random seed")
    p.add_argument("--sigma",        type=float, default=None, help="Override DP noise multiplier")
    p.add_argument("--selection",    type=str,   default="rl", choices=["rl", "random", "all"],
                   help="Client selection strategy")
    p.add_argument("--output-dir",   type=str,   default=config.RESULTS_DIR, help="Results directory")
    
    args = p.parse_args()
    set_seed(args.seed)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
