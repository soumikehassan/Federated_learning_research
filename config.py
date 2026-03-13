"""
config.py  ─ v2
All hyperparameters for the upgraded FL research framework.
New in v2:
  - SMPC / Secure Aggregation support
  - Multiple aggregation methods: FedAvg, FedProx, FedMedian, SCAFFOLD
  - Noise sensitivity experiments: sigma = 0.5, 1.0, 2.0
  - All results exported to CSV
"""

import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Dataset paths ──────────────────────────────────────────────────────────────
DATASET_PATHS = {
    "client_0": r"D:\Federate Learning Coding example\dataset\AugmentedAlzheimerDataset",
    "client_1": r"D:\Federate Learning Coding example\dataset\Ratinal_Deasis",
    "client_2": r"D:\Federate Learning Coding example\dataset\TB_Chest_Radiography_Database",
}

DATASET_NAMES = {
    "client_0": "Alzheimer MRI",
    "client_1": "Retinal Disease",
    "client_2": "TB Chest X-Ray",
}

NUM_CLASSES_PER_CLIENT = {
    "client_0": 4,   # MildDemented, ModerateDemented, NonDemented, VeryMildDemented
    "client_1": 7,   # Diabetic Retinopathy, Disc Edema, Healthy, Macular_Degeneration,
                     # Myopia, Retinal Detachment, Retinitis Pigmentosa
    "client_2": 2,   # Normal, Tuberculosis
}

# ── Model ──────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224
MODEL_SIZE   = "tiny"   # tiny | small
USE_PRETRAIN = True

# ── Federated Learning ─────────────────────────────────────────────────────────
NUM_CLIENTS   = 3
NUM_ROUNDS    = 20
LOCAL_EPOCHS  = 3
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4

# ── Aggregation methods to compare ────────────────────────────────────────────
# Each method is run and results saved separately for comparison
# Options: "FedAvg" | "FedProx" | "FedMedian" | "SCAFFOLD"
AGGREGATION_METHODS = ["FedAvg", "FedProx", "FedMedian", "SCAFFOLD"]
DEFAULT_AGGREGATION = "FedAvg"

# FedProx proximal term mu
FEDPROX_MU = 0.01

# ── Privacy mechanism ──────────────────────────────────────────────────────────
# Options: "DP" (Differential Privacy) | "SMPC" (Secure Multi-Party Computation)
#          "Both" (DP on gradients + SMPC on aggregation)
PRIVACY_MECHANISM = "DP"    # change to "SMPC" or "Both" as needed

# Differential Privacy
DP_ENABLED       = True
DP_NOISE_MULT    = 1.0      # sigma baseline
DP_MAX_GRAD_NORM = 1.0
DP_DELTA         = 1e-5

# Noise sensitivity experiment values (from doc: sigma = 0.5, 1.0, 2.0)
DP_NOISE_VARIANTS = [0.5, 1.0, 2.0]

# SMPC / Secure Aggregation
SMPC_ENABLED          = False   # set True to use secure aggregation
SMPC_NUM_SHARES       = 3       # number of secret shares
SMPC_RECONSTRUCTION_THRESHOLD = 2  # minimum shares to reconstruct

# ── Reinforcement Learning ─────────────────────────────────────────────────────
RL_HIDDEN_DIM    = 128
RL_LR            = 3e-4
RL_GAMMA         = 0.99
RL_EPSILON_START = 1.0
RL_EPSILON_END   = 0.1
RL_EPSILON_DECAY = 0.995
RL_BUFFER_SIZE   = 1000
RL_BATCH_SIZE    = 32
RL_MIN_CLIENTS   = 2

# ── CSV output ─────────────────────────────────────────────────────────────────
SAVE_CSV         = True     # save all metrics to CSV files
CSV_ROUND_LOG    = "round_metrics.csv"
CSV_CLIENT_LOG   = "client_test_metrics.csv"
CSV_PRIVACY_LOG  = "privacy_metrics.csv"
CSV_FAIRNESS_LOG = "fairness_metrics.csv"
CSV_RL_LOG       = "rl_metrics.csv"
CSV_COMPUTE_LOG  = "computational_metrics.csv"
CSV_COMPARE_LOG  = "method_comparison.csv"

# ── Misc ───────────────────────────────────────────────────────────────────────
SEED        = 42
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
NUM_WORKERS = 0
