# Dynamic Privacy-Preserving Federated Learning with Multimodal Swin Transformers and Reinforcement Learning for Collaborative Medical Diagnostics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Kaggle-lightgrey)

</div>

---

## Overview

This repository presents a novel **Dynamic Privacy-Preserving Federated Learning (DPPFL)** framework for collaborative medical image diagnostics. The system enables multiple hospitals to train a shared AI model **without sharing any patient data**, using:

- **Swin Transformer** — state-of-the-art vision backbone for medical image classification
- **Reinforcement Learning (DQN)** — intelligent client selection to maximize convergence
- **Differential Privacy (DP)** — mathematical privacy guarantee (ε < 10 for medical compliance)
- **Secure Multi-Party Computation (SMPC)** — cryptographic secure aggregation
- **4 Aggregation Methods** — FedAvg, FedProx, FedMedian, SCAFFOLD for comparison

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FEDERATED SERVER                          │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│   │  RL Agent   │  │  Aggregator  │  │  Privacy (DP/   │   │
│   │    (DQN)    │  │  FedAvg /    │  │    SMPC)        │   │
│   │ Client Sel. │  │  FedProx /   │  │  ε < 10 target  │   │
│   └─────────────┘  │  FedMedian / │  └─────────────────┘   │
│                    │  SCAFFOLD    │                          │
│                    └──────────────┘                          │
└────────────┬──────────────┬──────────────┬──────────────────┘
             │              │              │
    ┌────────▼───┐  ┌───────▼────┐  ┌─────▼──────┐
    │  Client 0  │  │  Client 1  │  │  Client 2  │
    │ Alzheimer  │  │  Retinal   │  │  TB X-Ray  │
    │  4 Classes │  │  7 Classes │  │  2 Classes │
    │ Swin-Tiny  │  │ Swin-Tiny  │  │ Swin-Tiny  │
    │  DP Noise  │  │  DP Noise  │  │  DP Noise  │
    └────────────┘  └────────────┘  └────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Heterogeneous FL** | 3 clients with different datasets and different number of classes |
| **Swin Transformer** | 27.5M parameter pretrained backbone (ImageNet weights) |
| **DQN Client Selection** | RL agent learns which clients to select each round |
| **Differential Privacy** | Gaussian mechanism — gradient clipping + noise injection |
| **SMPC Aggregation** | Additive secret sharing — server never sees individual updates |
| **4 Aggregation Methods** | FedAvg, FedProx, FedMedian, SCAFFOLD — all compared |
| **Full Metrics Suite** | 8 metric categories — classification, privacy, RL, FL, fairness, ablation |
| **CSV Export** | All results automatically saved to CSV for paper tables |
| **Publication Plots** | Training curves, confusion matrices, method comparison, fairness analysis |

---

## Datasets

| Client | Dataset | Classes | Task |
|--------|---------|---------|------|
| client_0 | [Augmented Alzheimer MRI](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset) | 4 | MildDemented, ModerateDemented, NonDemented, VeryMildDemented |
| client_1 | [Retinal Disease](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) | 7 | Diabetic Retinopathy, Disc Edema, Healthy, Macular Degeneration, Myopia, Retinal Detachment, Retinitis Pigmentosa |
| client_2 | [TB Chest X-Ray](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | 2 | Normal, Tuberculosis |

---

## Project Structure

```
fl_project_v2/
│
├── config.py                          # All hyperparameters and paths
├── main.py                            # Main training loop
├── feasibility_test.py                # Quick sanity check
│
├── models/
│   └── swin_transformer.py            # Swin Transformer builder (timm)
│
├── clients/
│   └── federated_client.py            # Local training + DP + metrics collection
│
├── server/
│   ├── federated_server.py            # Server aggregation + logging
│   └── aggregation_methods.py         # FedAvg, FedProx, FedMedian, SCAFFOLD
│
├── utils/
│   ├── differential_privacy.py        # Gaussian DP mechanism
│   ├── secure_aggregation.py          # SMPC / Secure Aggregation
│   ├── rl_client_selector.py          # DQN reinforcement learning agent
│   ├── evaluation_metrics.py          # All 8 metric categories + CSV logger
│   └── metrics.py                     # Plotting utilities
│
├── data/
│   └── dataset.py                     # Dataset loader (auto-discovers classes)
│
└── results/                           # Auto-generated results
    ├── FedAvg/
    │   ├── training_history.json
    │   ├── training_curves.png
    │   ├── classification_metrics_FedAvg.png
    │   ├── confusion_matrices_FedAvg.png
    │   └── best_model.pt
    ├── FedProx/
    ├── FedMedian/
    ├── SCAFFOLD/
    ├── method_comparison.png
    ├── fairness_analysis.png
    ├── round_metrics.csv
    ├── client_test_metrics.csv
    ├── privacy_metrics.csv
    ├── fairness_metrics.csv
    ├── rl_metrics.csv
    ├── fl_metrics.csv
    ├── computational_metrics.csv
    └── method_comparison.csv
```

---

## Installation

### Prerequisites
- Python 3.11
- Windows / Linux / macOS
- NVIDIA GPU (optional but recommended)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fl-medical-diagnostics.git
cd fl-medical-diagnostics/fl_project_v2

# 2. Create virtual environment
python -m venv fl_env

# 3. Activate (Windows)
.\fl_env\Scripts\Activate.ps1

# Activate (Linux/Mac)
source fl_env/bin/activate

# 4. Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch (GPU — CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Install all other dependencies
pip install timm scikit-learn matplotlib numpy pandas psutil
```

---

## Configuration

Update dataset paths in `config.py`:

```python
DATASET_PATHS = {
    "client_0": r"path/to/AugmentedAlzheimerDataset",
    "client_1": r"path/to/Retinal_Disease",
    "client_2": r"path/to/TB_Chest_Radiography_Database",
}
```

---

## Usage

### Quick sanity check
```bash
python feasibility_test.py
```

### Test run (fast — ~1 hour CPU)
```bash
python main.py --rounds 3 --local-epochs 1 --max-samples 50
```

### Single aggregation method
```bash
python main.py --rounds 20 --local-epochs 3 --method FedAvg
python main.py --rounds 20 --local-epochs 3 --method FedProx
python main.py --rounds 20 --local-epochs 3 --method FedMedian
python main.py --rounds 20 --local-epochs 3 --method SCAFFOLD
```

### All 4 methods compared
```bash
python main.py --rounds 20 --local-epochs 3
```

### With SMPC (Secure Aggregation)
```bash
python main.py --rounds 20 --local-epochs 3 --use-smpc
```

### Noise sensitivity experiment (sigma = 0.5, 1.0, 2.0)
```bash
python main.py --rounds 20 --local-epochs 3 --noise-sweep
```

### Disable Differential Privacy
```bash
python main.py --rounds 20 --local-epochs 3 --no-dp
```

### All CLI options
```
--rounds        Number of federated rounds (default: 20)
--local-epochs  Local training epochs per round (default: 3)
--batch-size    Batch size (default: 16)
--lr            Learning rate (default: 1e-4)
--model-size    Swin size: tiny | small | base (default: tiny)
--method        FedAvg | FedProx | FedMedian | SCAFFOLD | all
--no-dp         Disable differential privacy
--use-smpc      Enable secure aggregation
--no-pretrain   Train from scratch (no ImageNet weights)
--max-samples   Limit samples per class (for quick testing)
--noise-sweep   Run sigma = 0.5, 1.0, 2.0 comparison
```

---

## Evaluation Metrics

### 1. Classification Metrics (per client)
| Metric | Formula |
|--------|---------|
| Accuracy | Correct / Total |
| Precision | TP / (TP + FP) |
| Recall (Sensitivity) | TP / (TP + FN) |
| F1-Score | 2 × (P×R) / (P+R) |
| AUC-ROC | Area under ROC curve |
| Specificity | TN / (TN + FP) |
| Cohen's Kappa | Agreement beyond chance |
| Confusion Matrix | Per-class TP/FP/TN/FN |

### 2. Privacy Metrics
| Metric | Target |
|--------|--------|
| Privacy Budget (ε) | ε < 10 for medical |
| Privacy-Accuracy Tradeoff | Show the curve |
| Noise Multiplier Sensitivity | σ = 0.5, 1.0, 2.0 |
| Membership Inference AUC | Closer to 0.5 = better |

### 3. RL Metrics
- Cumulative reward, Selection frequency, Selection diversity entropy
- RL vs Random selection comparison, Convergence speed

### 4. FL Metrics
- Global convergence rate, Communication cost, Client drift
- Local vs Global accuracy gap, Rounds to convergence

### 5. Fairness Metrics
- Per-client accuracy variance, Worst-client performance
- Fairness gap (target < 0.15), Participation equity

### 6. Computational Metrics
- Training time per round, Total training time
- Model parameters (27.5M for Swin-Tiny), Memory usage, Inference time

---

## Demo Results (5 rounds, 1 epoch, test data)

| Method | Global Acc | Fairness Gap | Privacy ε |
|--------|-----------|-------------|-----------|
| **FedAvg** | **0.3333** | 0.4500 | 1.77 |
| FedProx | 0.2333 | 0.5000 | 1.77 |
| FedMedian | 0.2000 | 0.4000 | 1.20 |
| SCAFFOLD | 0.1818 | 0.4000 | 1.31 |

> Note: Results above use minimal test settings (2 rounds, 1 epoch, 50 samples).
> Full dataset results (20 rounds, 3 epochs) show significantly higher accuracy.

---

## Running on Kaggle GPU

1. Upload datasets to Kaggle Datasets
2. Create a new notebook → Settings → GPU T4
3. Add datasets and override paths:

```python
import sys
sys.path.insert(0, '/kaggle/input/fl-project-code')

import config
config.DATASET_PATHS = {
    "client_0": "/kaggle/input/alzheimer-fl/AugmentedAlzheimerDataset",
    "client_1": "/kaggle/input/retinal-fl/Ratinal_Deasis",
    "client_2": "/kaggle/input/tb-fl/TB_Chest_Radiography_Database",
}
config.RESULTS_DIR = "/kaggle/working/results"
config.NUM_WORKERS = 2
config.BATCH_SIZE  = 32
```

4. Run training:
```python
import subprocess
subprocess.run(["python", "main.py", "--rounds", "20", "--local-epochs", "3"])
```

**Expected time on Kaggle T4 GPU:** ~1.5 hours per method, ~6 hours for all 4 methods.

---

## Privacy Guarantee

The framework implements the **Gaussian Mechanism** for (ε, δ)-Differential Privacy:

```
ε = sqrt(2 × log(1.25/δ)) / σ × sqrt(T × q)
```

Where:
- `σ` = noise multiplier (default: 1.0)
- `T` = number of training steps
- `q` = sampling rate
- `δ` = 1e-5 (fixed)

**Target:** ε < 10 for medical research applications.

---

## Aggregation Methods

| Method | Paper | Key Idea |
|--------|-------|---------|
| **FedAvg** | McMahan et al. (2017) | Weighted average of client updates |
| **FedProx** | Li et al. (2020) | FedAvg + proximal term to reduce drift |
| **FedMedian** | Yin et al. (2018) | Coordinate-wise median — robust to outliers |
| **SCAFFOLD** | Karimireddy et al. (2020) | Control variates to fix client drift |

---

## Output Files

After training, results are saved in `results/`:

```
results/
├── method_comparison.csv        ← Main paper comparison table
├── client_test_metrics.csv      ← Per-client classification metrics
├── round_metrics.csv            ← Per-round training log
├── privacy_metrics.csv          ← Privacy budget tracking
├── fairness_metrics.csv         ← Fairness analysis
├── rl_metrics.csv               ← RL agent performance
├── fl_metrics.csv               ← FL convergence metrics
├── computational_metrics.csv    ← Time and memory usage
├── method_comparison.png        ← Bar chart comparing all methods
├── fairness_analysis.png        ← Fairness gap visualization
└── {method}/
    ├── training_curves.png      ← 6-panel training progress
    ├── classification_metrics_{method}.png
    ├── confusion_matrices_{method}.png
    ├── training_history.json
    ├── best_model.pt
    └── rl_agent.pt
```

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=1.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=2.0.0
psutil>=5.9.0
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dppfl2025,
  title   = {Dynamic Privacy-Preserving Federated Learning with Multimodal
             Swin Transformers and Reinforcement Learning for
             Collaborative Medical Diagnostics},
  author  = {Your Name},
  journal = {Journal Name},
  year    = {2025}
}
```

---

## References

1. McMahan et al. — *Communication-Efficient Learning of Deep Networks* (FedAvg, 2017)
2. Li et al. — *Federated Optimization in Heterogeneous Networks* (FedProx, 2020)
3. Yin et al. — *Byzantine-Robust Distributed Learning* (FedMedian, 2018)
4. Karimireddy et al. — *SCAFFOLD: Stochastic Controlled Averaging* (2020)
5. Liu et al. — *Swin Transformer: Hierarchical Vision Transformer* (2021)
6. Bonawitz et al. — *Practical Secure Aggregation for Privacy-Preserving ML* (SMPC, 2017)
7. Abadi et al. — *Deep Learning with Differential Privacy* (2016)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made for medical AI research — privacy-preserving collaborative learning
</div>
