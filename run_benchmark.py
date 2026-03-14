"""
run_benchmark.py
Automates the 12-run "Golden Path" strategy for ESWA submission.
- 3 Seeds x 3 Methods (RL, Random, All)
- 3 Noise Levels (Sigma Sweep)
- Statistical analysis (Mean, Std, T-test)
"""

import os
import subprocess
import pandas as pd
import numpy as np
import argparse
from scipy import stats

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10) # 10 for quick proof, use 20 for real paper
    parser.add_argument("--local-epochs", type=int, default=3)
    args = parser.parse_args()

    results_base = "results/benchmark"
    os.makedirs(results_base, exist_ok=True)

    seeds = [42 + i for i in range(args.seeds)]
    selection_modes = ["rl", "random", "all"]
    sigmas = [0.5, 1.0, 2.0]

    all_stats = []

    # Stage 1: Core Rigor (Multi-seed x Methods)
    print("\n[Stage 1] Core Rigor Benchmark (3 Seeds x 3 Methods)")
    for seed in seeds:
        for mode in selection_modes:
            out_dir = os.path.join(results_base, f"{mode}_seed{seed}")
            cmd = [
                "python3", "main.py",
                "--method", "FedAvg",
                "--selection", mode,
                "--seed", str(seed),
                "--rounds", str(args.rounds),
                "--local-epochs", str(args.local_epochs),
                "--output-dir", out_dir,
                "--max-samples", "100" # Use subset for speed during development
            ]
            run_cmd(cmd)
            
            # Collect final accuracy
            comp_path = f"{out_dir}/method_comparison.csv"
            if os.path.exists(comp_path):
                df = pd.read_csv(comp_path)
                final_acc = df.iloc[-1]["global_acc"]
                all_stats.append({
                    "seed": seed,
                    "mode": mode,
                    "accuracy": final_acc
                })

    # Save Stage 1 Report
    df_stage1 = pd.DataFrame(all_stats)
    df_stage1.to_csv(f"{results_base}/stage1_raw.csv", index=False)
    
    # Calculate Mean/Std
    summary = df_stage1.groupby("mode")["accuracy"].agg(["mean", "std"]).reset_index()
    print("\nTable 1: Final Accuracy (Mean ± Std)")
    print(summary)
    summary.to_csv(f"{results_base}/table1_accuracy.csv", index=False)

    # Stage 2: Privacy Sweep (Seed 42)
    print("\n[Stage 2] Privacy-Accuracy Tradeoff (Sigma Sweep)")
    privacy_stats = []
    for sigma in sigmas:
        out_dir = os.path.join(results_base, f"rl_sigma{sigma}")
        cmd = [
            "python3", "main.py",
            "--method", "FedAvg",
            "--selection", "rl",
            "--seed", "42",
            "--sigma", str(sigma),
            "--rounds", str(args.rounds),
            "--output-dir", out_dir,
            "--max-samples", "100"
        ]
        run_cmd(cmd)
        
        comp_path = f"{out_dir}/method_comparison.csv"
        if os.path.exists(comp_path):
            df = pd.read_csv(comp_path)
            privacy_stats.append({
                "sigma": sigma,
                "accuracy": df.iloc[-1]["global_acc"],
                "epsilon": df.iloc[-1]["final_epsilon"] if "final_epsilon" in df.columns else 0.0
            })

    df_stage2 = pd.DataFrame(privacy_stats)
    print("\nTable 2: Privacy-Accuracy Tradeoff")
    print(df_stage2)
    df_stage2.to_csv(f"{results_base}/table2_privacy.csv", index=False)

    print(f"\nBenchmark completed. Results saved to {results_base}/")

if __name__ == "__main__":
    main()
