# Strategic Enhancement for Expert Systems with Applications (ESWA) Submission

This plan outlines the systematic restructuring of the **Dynamic Privacy-Preserving Federated Learning (DPPFL)** framework to meet the rigorous standards of the ESWA journal. The core pivot is moving from a "multi-task" setup to a "controlled single-task" setup to properly benchmark the RL-based system.

## Core Objectives
1.  **System Validation**: Propose a robust system where **DP + RL** mitigates the performance drop caused by **Non-IID (Label Skew)** data.
2.  **Controlled Heterogeneity**: Implement a system-driven data partitioning mechanism (Dirichlet) to create measurable IID vs. Non-IID environments.
3.  **Statistical Significance**: Move from single-run demonstrations to multi-seed experiments with mean/standard deviation reporting.

## Proposed Technical Enhancements

### 1. Unified Dataset & Dirichlet Partitioner [PIVOT]
We will transition from using 3 different datasets to using **one primary dataset** (e.g., Alzheimer MRI) partitioned across clients.
- **New Component**: `utils/data_partitioner.py` to handle IID and Non-IID (Dirichlet $\alpha$) splits.
- **Update**: [data/dataset.py](file:///media/ayon1901/SERVER/Federated_learning_research/data/dataset.py) to integrate with the partitioner.
- **Benefit**: Allows a direct "Apple-to-Apple" IID vs Non-IID comparison.

### 2. RL-Agent Awareness of Skew
The RL agent needs to "sense" the Non-IIDness to make better decisions.
- **Enhancement**: Add `class_distribution` aware rewards to the RL agent.
- **Target**: [utils/rl_client_selector.py](file:///media/ayon1901/SERVER/Federated_learning_research/utils/rl_client_selector.py) implementation.

### 3. Statistical Rigor & Reporting
- **Experimental Script**: `run_benchmark.py` will execute 5 runs for each configuration (IID, Non-IID $\alpha=0.1, 0.5$, etc.) with different seeds.
- **Analysis**: Automate T-tests to prove the RL selection is statistically better than random selection.

## Verification Plan

### Automated Experiments
- **Ablation Study**: Compare `Random Selection` vs `RL Selection` vs `Greedy Selection` across the IID/Non-IID spectrum.
- **Privacy Sensitivity**: Demonstrate ε stability across multiple runs.

### Manual Review
- [ ] Verify that all CSV logs in `results/` contain the new RL state features for verification in paper tables.
