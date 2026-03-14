# Experiments

This folder contains all experiments for the paper:

**"Dynamic Privacy-Preserving Federated Learning with Reinforcement Learning-Based Client Selection"**

---

## Overview

We conduct 3 experiments to validate our framework:

| # | Experiment | Purpose | Key Figures |
|---|------------|---------|-------------|
| 1 | [RL Effectiveness](1_rl_effectiveness/) | Prove RL selection outperforms baselines, especially under non-IID | Convergence curves, Alpha sweep |
| 2 | [Privacy Tradeoff](2_privacy_tradeoff/) | Prove DP provides privacy (ε < 10) with bounded accuracy loss | Privacy-accuracy curve |
| 3 | [Component Ablation](3_component_ablation/) | Prove each component (RL, DP, Fairness) contributes positively | Ablation bar chart |

---

## Quick Start

### Run All Experiments
```bash
python run_all_experiments.py --seeds 3 --rounds 20
```

### Run Individual Experiment
```bash
# Experiment 1: RL Effectiveness
cd experiments/1_rl_effectiveness
python experiment.py --seeds 3 --rounds 20

# Experiment 2: Privacy Tradeoff
cd experiments/2_privacy_tradeoff
python experiment.py --rounds 20

# Experiment 3: Component Ablation
cd experiments/3_component_ablation
python experiment.py --seeds 3 --rounds 20
```

---

## Paper Structure

The experiments map to paper sections:

```
Section 5: Experiments
├── 5.1 Experimental Setup
├── 5.2 RL Effectiveness (Experiment 1)
│   ├── 5.2.1 Comparison with Baselines
│   └── 5.2.2 Non-IID Robustness (Alpha Sweep)
├── 5.3 Privacy Analysis (Experiment 2)
└── 5.4 Ablation Study (Experiment 3)
```

---

## Output Structure

Each experiment generates:
- `figures/` - Publication-ready plots (PNG, 300 DPI)
- `tables/` - CSV files for paper tables
- Console output with pass/fail for hypothesis

---

## Total Experiments Required

| Experiment | Configurations | Seeds | Total Runs |
|------------|---------------|-------|------------|
| 1a: RL vs Baselines | 3 (RL, Random, All) | 3 | 9 |
| 1b: Alpha Sweep | 4 alphas × 2 methods | 1 | 8 |
| 2: Privacy | 3 sigmas | 1 | 3 |
| 3: Ablation | 4 configs (Full, No-RL, No-DP, No-Fair) | 3 | 12 |
| **Total** | | | **32 runs** |

With `--max-samples 100`, total runtime ~2-3 hours on GPU.
