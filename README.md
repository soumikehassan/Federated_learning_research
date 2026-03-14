# Privacy-Preserving Federated Learning for Medical Expert Systems: RL-Based Client Selection with Differential Privacy

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Kaggle-lightgrey)
![Domain](https://img.shields.io/badge/Domain-Medical%20AI-red)

</div>

---

## Overview

This repository implements a **Privacy-Preserving Federated Learning Framework** for **Medical Expert Systems**. The framework enables hospitals and clinical institutions to collaboratively train diagnostic AI models **without sharing patient data**, addressing critical privacy requirements in healthcare (HIPAA, GDPR compliance).

### Medical Expert System Context

Modern medical expert systems require large, diverse datasets to achieve clinical-grade performance. However, patient data privacy regulations prevent hospitals from sharing medical images directly. This framework solves this problem through:

| Challenge | Our Solution |
|-----------|--------------|
| Patient data cannot leave hospitals | **Federated Learning** — model travels, not data |
| Privacy regulations (HIPAA/GDPR) | **Differential Privacy** — mathematical privacy guarantee |
| Heterogeneous hospital data | **RL-based client selection** — intelligently select training partners |
| Cross-specialty collaboration | **Multi-domain support** — neurology, ophthalmology, pulmonology |

### Clinical Applications

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MEDICAL EXPERT SYSTEM                               │
│                                                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│   │   NEUROLOGY     │  │  OPHTHALMOLOGY  │  │  PULMONOLOGY    │        │
│   │                 │  │                 │  │                 │        │
│   │ Alzheimer's     │  │ Diabetic        │  │ Tuberculosis    │        │
│   │ Dementia        │  │ Retinopathy     │  │ Detection       │        │
│   │ Detection       │  │ Macular Degen.  │  │                 │        │
│   │                 │  │ Retinal Detach. │  │                 │        │
│   │ Brain MRI       │  │ Fundus Images   │  │ Chest X-Ray     │        │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘        │
│            │                    │                    │                  │
│            └────────────────────┼────────────────────┘                  │
│                                 │                                       │
│                    ┌────────────▼────────────┐                         │
│                    │   FEDERATED SERVER      │                         │
│                    │   (No raw patient data) │                         │
│                    │   • RL Client Selector  │                         │
│                    │   • DP Privacy Guard    │                         │
│                    │   • Model Aggregator    │                         │
│                    └─────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Contributions for Medical Expert Systems

| Contribution | Description | Relevance to Expert Systems |
|--------------|-------------|----------------------------|
| **Privacy-Preserving FL** | DP + SMPC for HIPAA/GDPR compliance | Enables clinical deployment |
| **RL Client Selection** | DQN learns optimal hospital participation | Maximizes knowledge transfer |
| **Cross-Domain Learning** | Neurology + Ophthalmology + Pulmonology | Multi-specialty expert system |
| **Non-IID Robustness** | Dirichlet partitioning for realistic data skew | Handles hospital heterogeneity |
| **Task Fairness** | Ensures no medical domain is left behind | Balanced diagnostic capability |

---

## Medical Datasets

The framework is validated on three clinically relevant medical imaging tasks:

| Clinical Domain | Dataset | Classes | Diagnostic Task | Modality |
|-----------------|---------|---------|-----------------|----------|
| **Neurology** | [Augmented Alzheimer MRI](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset) | 4 | MildDemented, ModerateDemented, NonDemented, VeryMildDemented | MRI |
| **Ophthalmology** | [Retinal Disease](https://www.kaggle.com/datasets/alemranp/ratinal-deasis) | 7 | Diabetic Retinopathy, Disc Edema, Macular Degeneration, Myopia, Retinal Detachment, Retinitis Pigmentosa, Healthy | Fundus |
| **Pulmonology** | [TB Chest X-Ray](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | 2 | Normal, Tuberculosis | X-Ray |

### Why These Datasets?

- **Neurodegenerative disease**: Early Alzheimer's detection from brain scans
- **Diabetic complications**: Leading cause of blindness in diabetic patients
- **Infectious disease**: TB affects 10M+ people annually worldwide

---

## Architecture

### System Components

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
    │ Hospital A │  │ Hospital B │  │ Hospital C │
    │ Neurology  │  │Ophthalmol. │  │Pulmonology │
    │ Swin-Tiny  │  │ Swin-Tiny  │  │ Swin-Tiny  │
    │  DP Noise  │  │  DP Noise  │  │  DP Noise  │
    └────────────┘  └────────────┘  └────────────┘
```

### Privacy Guarantees for Medical Data

| Mechanism | Protection Level | Medical Compliance |
|-----------|-----------------|-------------------|
| **Differential Privacy** | ε < 10 | HIPAA Safe Harbor |
| **Secure Aggregation (SMPC)** | Server never sees raw updates | GDPR Article 32 |
| **Federated Learning** | Data never leaves hospital | Data minimization principle |

---

## Project Structure

```
Federated_learning_research/
│
├── config.py                          # All hyperparameters and paths
├── main.py                            # Main training loop
├── run_all_experiments.py             # Run all paper experiments
├── feasibility_test.py                # Quick sanity check
├── requirements.txt                   # pip install -r requirements.txt
│
├── notebooks/                         # Colab: run on Google GPU
│   └── colab_run_experiments.ipynb   # Open in Colab → Runtime → GPU → Run all
│
├── experiments/                       # Paper experiments (NEW)
│   ├── README.md                      # Experiment overview
│   ├── 1_rl_effectiveness/            # RL vs baselines
│   ├── 2_privacy_tradeoff/            # DP analysis
│   └── 3_component_ablation/          # Ablation study
│
├── models/
│   └── swin_transformer.py            # Swin Transformer backbone
│
├── clients/
│   └── federated_client.py            # Local training + DP
│
├── server/
│   ├── federated_server.py            # Aggregation + logging
│   └── aggregation_methods.py         # FedAvg, FedProx, FedMedian, SCAFFOLD
│
├── utils/
│   ├── differential_privacy.py        # Gaussian DP mechanism
│   ├── secure_aggregation.py          # SMPC implementation
│   ├── rl_client_selector.py          # DQN agent
│   ├── data_partitioner.py            # Dirichlet non-IID partitioning
│   ├── evaluation_metrics.py          # 8 metric categories
│   └── metrics.py                     # Plotting utilities
│
├── data/
│   └── dataset.py                     # Medical image dataset loader
│
└── results/                           # Auto-generated results
```

---

## Installation

### Prerequisites
- Python 3.11
- NVIDIA GPU (recommended for medical image processing)
- 8GB+ RAM

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Federated_learning_research.git
cd Federated_learning_research

# 2. Create virtual environment
python -m venv fl_env
source fl_env/bin/activate  # Linux/Mac
# or: .\fl_env\Scripts\Activate.ps1  # Windows

# 3. Install PyTorch (GPU — CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install timm scikit-learn matplotlib numpy pandas psutil scipy
```

---

## Configuration

Update dataset paths in `config.py`:

```python
DATASET_PATHS = {
    "client_0_0": r"path/to/AugmentedAlzheimerDataset",  # Neurology
    "client_0_1": r"path/to/AugmentedAlzheimerDataset",
    "client_1_0": r"path/to/Retinal_Disease",            # Ophthalmology
    "client_1_1": r"path/to/Retinal_Disease",
    "client_2_0": r"path/to/TB_Chest_Radiography",       # Pulmonology
    "client_2_1": r"path/to/TB_Chest_Radiography",
}
```

---

## Usage

### Quick Test (sanity check)
```bash
python feasibility_test.py
```

### Run Single Experiment
```bash
# Test run (fast)
python main.py --rounds 3 --local-epochs 1 --max-samples 50

# Full training
python main.py --rounds 20 --local-epochs 3 --method FedAvg

# With different client selection
python main.py --selection rl      # RL-based (default)
python main.py --selection random  # Random selection
python main.py --selection all     # All clients (baseline)
```

### Run All Paper Experiments
```bash
python run_all_experiments.py --seeds 3 --rounds 20
```

### CLI Options
```
--rounds        Federated rounds (default: 20)
--local-epochs  Local training epochs (default: 3)
--batch-size    Batch size (default: 16)
--lr            Learning rate (default: 1e-4)
--method        FedAvg | FedProx | FedMedian | SCAFFOLD | all
--selection     rl | random | all
--no-dp         Disable differential privacy
--use-smpc      Enable secure aggregation
--noise-sweep   Run sigma = 0.5, 1.0, 2.0 comparison
--max-samples   Limit samples per class (for quick testing)
```

---

## Experiments (for Paper)

The `experiments/` folder contains 3 structured experiments:

| # | Experiment | Medical Relevance | Hypothesis |
|---|------------|-------------------|------------|
| 1 | [RL Effectiveness](experiments/1_rl_effectiveness/) | Optimal hospital selection for diagnosis | RL outperforms random, especially under data heterogeneity |
| 2 | [Privacy Tradeoff](experiments/2_privacy_tradeoff/) | HIPAA/GDPR compliance vs accuracy | DP provides ε < 10 with < 5% accuracy loss |
| 3 | [Component Ablation](experiments/3_component_ablation/) | System design validation | Each component contributes to medical diagnosis |

Each experiment has:
- `experiment.py` — runnable script
- `README.md` — hypothesis + "if proven" / "if not proven" guidance
- `figures/` — publication-ready plots

---

## Evaluation Metrics

### Medical Diagnostic Metrics
| Metric | Clinical Interpretation |
|--------|------------------------|
| **Sensitivity (Recall)** | Ability to detect disease (minimize false negatives) |
| **Specificity** | Ability to identify healthy cases (minimize false positives) |
| **AUC-ROC** | Overall diagnostic discrimination ability |
| **F1-Score** | Balance between precision and recall |

### Privacy Metrics
| Metric | Medical Requirement |
|--------|---------------------|
| **Privacy Budget (ε)** | ε < 10 for HIPAA compliance |
| **MIA Resistance** | Protect against membership inference attacks |

### Federated Learning Metrics
| Metric | System Performance |
|--------|-------------------|
| **Fairness Gap** | Equal diagnostic capability across specialties |
| **Convergence Speed** | Rounds until clinical-grade accuracy |
| **Communication Cost** | Bandwidth requirements for hospital networks |

---

## Results Summary

### Medical Diagnostic Performance

| Clinical Domain | Full System | No RL | No DP | No Fairness |
|-----------------|-------------|-------|-------|-------------|
| **Neurology (Alzheimer)** | TBD | TBD | TBD | TBD |
| **Ophthalmology (Retinal)** | TBD | TBD | TBD | TBD |
| **Pulmonology (TB)** | TBD | TBD | TBD | TBD |
| **Overall** | TBD | TBD | TBD | TBD |

*Results will be populated after running experiments.*

---

## Running on Kaggle GPU

For hospitals without local GPU infrastructure:

1. Upload datasets to Kaggle Datasets
2. Create notebook → Settings → GPU T4
3. Override paths:

```python
import sys, config
sys.path.insert(0, '/kaggle/input/fl-project-code')

config.DATASET_PATHS = {
    "client_0_0": "/kaggle/input/alzheimer-fl/AugmentedAlzheimerDataset",
    "client_1_0": "/kaggle/input/retinal-fl/Ratinal_Deasis",
    "client_2_0": "/kaggle/input/tb-fl/TB_Chest_Radiography_Database",
    # ... (add all 6 clients)
}
config.RESULTS_DIR = "/kaggle/working/results"
```

---

## Running on Google Colab GPU (from Cursor)

You can run experiments on **Google Colab’s free GPU** while keeping your code in Cursor.

### 1. One-click notebook (recommended)

1. **Push your repo to GitHub** (if you haven’t already).
2. **Open the Colab notebook** in your browser:
   - Replace `YOUR_USERNAME` with your GitHub username and open:
   - **https://colab.research.google.com/github/YOUR_USERNAME/Federated_learning_research/blob/main/notebooks/colab_run_experiments.ipynb**
   - Or in Cursor: right-click `notebooks/colab_run_experiments.ipynb` → **Open in Colab** (if you have the Colab extension).
3. In Colab: **Runtime → Change runtime type → GPU (e.g. T4)**.
4. In the notebook, set `REPO_URL` to your repo (e.g. `https://github.com/YOUR_USERNAME/Federated_learning_research.git`).
5. Run all cells. The notebook will clone the repo, install dependencies, optionally download datasets from Kaggle, set `COLAB_DATASET_ROOT` / `COLAB_RESULTS_DIR`, and run `main.py`.

Datasets:

- **Option A:** Use Kaggle: add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab **Secrets** (🔑 in the left sidebar), and set `USE_KAGGLE = True` in the notebook.
- **Option B:** Upload the three dataset folders to Google Drive under e.g. `My Drive/datasets/`, set `USE_DRIVE = True`, and set `DATASET_ROOT` to `/content/drive/MyDrive/datasets`.

Results are written to the path in `COLAB_RESULTS_DIR` (e.g. `/content/results` or, with Drive, `/content/drive/MyDrive/fl_results`).

### 2. Workflow: Cursor + Colab

| In Cursor | In Colab |
|-----------|----------|
| Edit code, run locally (no GPU or small tests) | Run full experiments on GPU via the notebook |
| Push to GitHub | Notebook clones/pulls and runs `main.py` or `run_all_experiments.py` |

No need to copy‑paste code: point the notebook at your repo and run.

---

## Privacy Guarantee

The framework implements **(ε, δ)-Differential Privacy** suitable for medical data:

```
ε = √(2 × log(1.25/δ)) / σ × √(T × q)
```

| Parameter | Value | Medical Justification |
|-----------|-------|----------------------|
| σ (noise) | 1.0 | Balanced privacy-utility |
| δ (failure prob) | 1e-5 | Negligible breach probability |
| **Target ε** | **< 10** | HIPAA Safe Harbor compliant |

---

## Aggregation Methods

| Method | Reference | Medical FL Benefit |
|--------|-----------|-------------------|
| **FedAvg** | McMahan et al. (2017) | Standard baseline |
| **FedProx** | Li et al. (2020) | Handles heterogeneous hospital data |
| **FedMedian** | Yin et al. (2018) | Robust to outlier hospitals |
| **SCAFFOLD** | Karimireddy et al. (2020) | Corrects hospital-specific drift |

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
scipy>=1.10.0
```

---

## Citation

If you use this code for medical AI research, please cite:

```bibtex
@article{dppfl2025,
  title   = {Privacy-Preserving Federated Learning with Reinforcement Learning
             for Medical Expert Systems: A Multi-Domain Diagnostic Framework},
  author  = {Your Name},
  journal = {Expert Systems with Applications},
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
6. Abadi et al. — *Deep Learning with Differential Privacy* (2016)
7. Rieke et al. — *The Future of Digital Health with Federated Learning* (2020)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for Medical Expert Systems Research**

Privacy-Preserving | Federated | Clinically Relevant

</div>
