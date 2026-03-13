"""
feasibility_test.py
Run this FIRST to verify every component works on your machine.
Uses 100% synthetic data — no dataset needed.
Expected runtime: ~60-120 seconds on CPU.

Usage:
    python feasibility_test.py
"""

import sys
import os
import time
import torch
import numpy as np

# ── Make sure Python can find our subfolders ──────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

print("\n" + "="*65)
print("  FEASIBILITY TEST — Dynamic FL + Swin + RL + DP")
print("="*65)

PASSED, FAILED = [], []

def test(name, fn):
    try:
        result = fn()
        label  = f" → {result}" if result else ""
        print(f"  ✓ {name}{label}")
        PASSED.append(name)
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        FAILED.append(name)
        return False

# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Environment")
test("PyTorch",    lambda: f"v{torch.__version__}")
test("Device",     lambda: f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
try:
    import timm
    test("timm",   lambda: f"v{timm.__version__}")
    HAS_TIMM = True
except ImportError:
    print("  ⚠ timm not installed — lightweight fallback will be used")
    HAS_TIMM = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Synthetic Datasets")
from torch.utils.data import TensorDataset, DataLoader

def make_data(num_classes, n=160, img=64):
    x = torch.randn(n, 3, img, img)
    y = torch.randint(0, num_classes, (n,))
    return {
        "train": DataLoader(TensorDataset(x[:120], y[:120]), batch_size=16, shuffle=True),
        "val":   DataLoader(TensorDataset(x[120:140], y[120:140]), batch_size=16),
        "test":  DataLoader(TensorDataset(x[140:], y[140:]), batch_size=16),
        "num_classes": num_classes,
        "classes": [f"class_{i}" for i in range(num_classes)],
        "dataset_size": 120,
    }

cdata = {"client_0": make_data(4), "client_1": make_data(4), "client_2": make_data(2)}
test("Create 3 client datasets", lambda: "3 clients OK")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Swin Transformer Models")
from models.swin_transformer import build_swin_model, LightweightSwinTransformer

def chk_model(nc):
    # pretrained=False avoids any internet download during testing
    # Using LightweightSwinTransformer directly for speed + no download
    m = LightweightSwinTransformer(
        img_size=64, embed_dim=48, depths=[1, 1],
        num_heads=[3, 6], num_classes=nc
    ).to(device).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64).to(device))
    assert out.shape == (2, nc), f"Expected (2,{nc}), got {out.shape}"
    p = sum(x.numel() for x in m.parameters()) / 1e6
    return f"{p:.2f}M params, output shape={out.shape}"

test("client_0 model (4 classes — Alzheimer)",  lambda: chk_model(4))
test("client_1 model (7 classes — Retinal)",    lambda: chk_model(7))
test("client_2 model (2 classes — TB X-Ray)",   lambda: chk_model(2))

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Differential Privacy")
from utils.differential_privacy import DPMechanism

def chk_dp():
    dp = DPMechanism(noise_multiplier=1.0, max_grad_norm=1.0, enabled=True)
    m  = LightweightSwinTransformer(img_size=64, embed_dim=32, depths=[1,1],
                                    num_heads=[1,2], num_classes=2).to(device)
    loss = torch.nn.CrossEntropyLoss()(
        m(torch.randn(2,3,64,64).to(device)),
        torch.randint(0,2,(2,)).to(device)
    )
    loss.backward()
    dp.clip_gradients(m)
    dp.add_noise_to_gradients(m)
    r = dp.get_privacy_report(10, 0.1)
    return f"ε={r['epsilon']:.4f}"

test("Gradient clip + noise",     chk_dp)
test("Privatize model update",    lambda: DPMechanism(enabled=True).privatize_model_update(
    {"w": torch.randn(4,4)}) is not None and "OK")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] RL Client Selector")
from utils.rl_client_selector import RLClientSelector

def chk_rl():
    rl = RLClientSelector(num_clients=3, hidden_dim=64, min_clients=2, device=str(device))
    stats = {
        "client_0": {"loss": 0.8, "accuracy": 0.65, "data_size": 120},
        "client_1": {"loss": 1.2, "accuracy": 0.55, "data_size": 120},
        "client_2": {"loss": 0.5, "accuracy": 0.75, "data_size": 120},
    }
    sel = rl.select_clients(stats, 1)
    assert len(sel) >= 2
    st  = rl.build_state(stats)
    for _ in range(40):
        rl.store_transition(st, sel, 0.1, st, False)
    loss = rl.update()
    rwd  = rl.compute_reward(0.6, 0.65, sel, 2)
    return f"selected={sel}, loss={loss:.6f}, reward={rwd:.4f}"

test("DQN client selection", chk_rl)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Federated Client (Local Training)")
from clients.federated_client import FederatedClient

def chk_client():
    m  = LightweightSwinTransformer(img_size=64, embed_dim=32, depths=[1,1],
                                    num_heads=[1,2], num_classes=4).to(device)
    dp = DPMechanism(noise_multiplier=1.0, max_grad_norm=1.0, enabled=True)
    cl = FederatedClient("client_0", m, make_data(4), dp, device, lr=1e-4, local_epochs=1)
    upd = cl.train_local()
    assert "weight_delta" in upd and "train_loss" in upd
    return f"loss={upd['train_loss']:.4f}, acc={upd['val_accuracy']:.4f}"

test("Client local train + DP", chk_client)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Federated Server (FedAvg)")
from server.federated_server import FederatedServer

def chk_server():
    m  = LightweightSwinTransformer(img_size=64, embed_dim=32, depths=[1,1],
                                    num_heads=[1,2], num_classes=4).to(device)
    dp = DPMechanism(enabled=False)
    sv = FederatedServer(m, dp, device, results_dir="results")
    upds = [
        {"weight_delta": {k: torch.zeros_like(v) for k,v in m.state_dict().items()},
         "train_loss": 0.8, "val_accuracy": 0.6, "data_size": 120},
        {"weight_delta": {k: torch.zeros_like(v) for k,v in m.state_dict().items()},
         "train_loss": 0.6, "val_accuracy": 0.7, "data_size": 100},
    ]
    w = sv.aggregate(upds, ["client_0","client_1"])
    assert w is not None
    return "FedAvg OK"

test("Server FedAvg aggregation", chk_server)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] End-to-End Mini Run (3 rounds, synthetic data)")

def chk_e2e():
    nc_map = {"client_0": 4, "client_1": 7, "client_2": 2}  # real class counts
    cdata_e2e = {
        "client_0": make_data(4),
        "client_1": make_data(7),
        "client_2": make_data(2),
    }
    models = {
        cid: LightweightSwinTransformer(img_size=64, embed_dim=32, depths=[1,1],
                                         num_heads=[1,2], num_classes=nc).to(device)
        for cid, nc in nc_map.items()
    }
    dp_map = {cid: DPMechanism(noise_multiplier=0.5, max_grad_norm=1.0) for cid in models}
    clients = {
        cid: FederatedClient(cid, models[cid], cdata_e2e[cid], dp_map[cid], device, lr=1e-4, local_epochs=1)
        for cid in models
    }
    rl = RLClientSelector(num_clients=3, hidden_dim=64, min_clients=2, device=str(device))

    prev_acc, prev_st, prev_sel = 0.0, None, None
    for r in range(1, 4):
        stats   = {cid: c.get_rl_state_features() for cid, c in clients.items()}
        state   = rl.build_state(stats)
        sel_idx = rl.select_clients(stats, r)
        sel_cid = [f"client_{i}" for i in sel_idx]
        updates = [clients[cid].train_local() for cid in sel_cid]
        acc     = np.mean([u["val_accuracy"] for u in updates])
        rwd     = rl.compute_reward(prev_acc, acc, sel_idx, r)
        nxt_st  = rl.build_state({cid: c.get_rl_state_features() for cid, c in clients.items()})
        if prev_st is not None:
            rl.store_transition(prev_st, prev_sel, rwd, state, False)
        rl.update()
        prev_acc, prev_st, prev_sel = acc, state, sel_idx

    return f"3 rounds done, final_acc={acc:.4f}"

t0 = time.time()
test("End-to-end 3-round run", chk_e2e)
print(f"  (took {time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"  RESULTS: {len(PASSED)} passed  |  {len(FAILED)} failed")
if FAILED:
    print(f"\n  Failed:")
    for f in FAILED:
        print(f"    ✗ {f}")
    print("\n  Fix these before running main.py")
else:
    print("\n  ✓ All tests passed! Run full training:")
    print("    python main.py --rounds 5 --local-epochs 1 --max-samples 50")
    print("    python main.py --rounds 20 --local-epochs 3")
print("="*65 + "\n")
