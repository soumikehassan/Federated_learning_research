"""
run_all_experiments.py
Master script to run all 3 experiments for the paper.

Runs:
1. RL Effectiveness (baseline comparison + alpha sweep)
2. Privacy Tradeoff (DP noise sweep)
3. Component Ablation (No-RL, No-DP, No-Fairness)

Usage:
    python run_all_experiments.py --seeds 3 --rounds 20
    python run_all_experiments.py --seeds 1 --rounds 5 --max-samples 50  # quick
"""

import sys
import os
import argparse
import subprocess
import json
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(THIS_DIR, "experiments")
RESULTS_DIR = os.path.join(THIS_DIR, "results", "benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment(name, script_path, args_list):
    """Run a single experiment and capture results."""
    print(f"\n{'#'*60}")
    print(f"  Running: {name}")
    print(f"{'#'*60}")

    cmd = [sys.executable, script_path] + args_list
    print(f"  Command: {' '.join(cmd)}")

    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return {"name": name, "status": "FAILED", "duration": duration, "error": result.stderr[-500:]}

    # Extract hypothesis check from output
    output = result.stdout
    hypothesis_status = "UNKNOWN"
    if "HYPOTHESIS PROVEN" in output:
        hypothesis_status = "PROVEN"
    elif "HYPOTHESIS NOT PROVEN" in output:
        hypothesis_status = "NOT PROVEN"
    elif "HYPOTHESIS PARTIALLY PROVEN" in output:
        hypothesis_status = "PARTIALLY PROVEN"

    print(f"  Status: {hypothesis_status}")
    print(f"  Duration: {duration:.1f}s")

    return {
        "name": name,
        "status": "COMPLETED",
        "hypothesis": hypothesis_status,
        "duration": duration,
    }


def main(args):
    print("\n" + "="*60)
    print("  RUNNING ALL EXPERIMENTS")
    print(f"  Seeds: {args.seeds}, Rounds: {args.rounds}")
    print("="*60)

    # Build common args
    common_args = [
        "--seeds", str(args.seeds),
        "--rounds", str(args.rounds),
        "--local-epochs", str(args.local_epochs),
    ]
    if args.max_samples:
        common_args.extend(["--max-samples", str(args.max_samples)])
    if args.no_pretrain:
        common_args.append("--no-pretrain")

    experiments = [
        {
            "name": "Experiment 1: RL Effectiveness",
            "script": os.path.join(EXPERIMENTS_DIR, "1_rl_effectiveness", "experiment.py"),
            "args": common_args,
        },
        {
            "name": "Experiment 2: Privacy Tradeoff",
            "script": os.path.join(EXPERIMENTS_DIR, "2_privacy_tradeoff", "experiment.py"),
            "args": ["--rounds", str(args.rounds), "--local-epochs", str(args.local_epochs)] +
                    (["--max-samples", str(args.max_samples)] if args.max_samples else []),
        },
        {
            "name": "Experiment 3: Component Ablation",
            "script": os.path.join(EXPERIMENTS_DIR, "3_component_ablation", "experiment.py"),
            "args": common_args,
        },
    ]

    all_results = []
    for exp in experiments:
        result = run_experiment(exp["name"], exp["script"], exp["args"])
        all_results.append(result)

    # Generate summary
    print("\n" + "="*60)
    print("  EXPERIMENT SUMMARY")
    print("="*60)

    summary_table = []
    for r in all_results:
        summary_table.append({
            "Experiment": r["name"],
            "Status": r["status"],
            "Hypothesis": r.get("hypothesis", "N/A"),
            "Duration (s)": f"{r['duration']:.1f}",
        })

    # Print table
    print(f"\n  {'Experiment':<35} {'Status':<12} {'Hypothesis':<20} {'Time':<10}")
    print(f"  {'-'*77}")
    for row in summary_table:
        print(f"  {row['Experiment']:<35} {row['Status']:<12} {row['Hypothesis']:<20} {row['Duration (s)']:<10}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "all_experiments_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    # Paper summary
    print("\n" + "="*60)
    print("  PAPER SUMMARY")
    print("="*60)
    print("\n  Based on results, your paper should emphasize:")

    proven = [r for r in all_results if r.get("hypothesis") == "PROVEN"]
    not_proven = [r for r in all_results if r.get("hypothesis") == "NOT PROVEN"]

    if proven:
        print(f"\n  ✓ STRONG POINTS ({len(proven)} hypotheses proven):")
        for r in proven:
            print(f"    - {r['name']}")

    if not_proven:
        print(f"\n  ⚠ WEAK POINTS ({len(not_proven)} hypotheses not proven):")
        for r in not_proven:
            print(f"    - {r['name']}")
            print(f"      → Reframe or de-emphasize in paper")

    print(f"\n  All figures saved in: experiments/*/figures/")
    print(f"  All tables saved in: experiments/*/tables/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    p.add_argument("--rounds", type=int, default=20, help="Federated rounds")
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples per class")
    p.add_argument("--no-pretrain", action="store_true")
    args = p.parse_args()
    main(args)
