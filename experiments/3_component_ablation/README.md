# Experiment 3: Component Ablation

## Hypothesis

**Each component (RL selection, DP, Fairness reward) contributes positively to overall system performance. Removing any component degrades accuracy or fairness.**

---

## Setup

| Parameter | Value |
|-----------|-------|
| Datasets | Alzheimer MRI, Retinal Disease, TB Chest X-Ray |
| Clients | 6 (2 per domain) |
| Rounds | 20 |
| Seeds | 3 (42, 43, 44) |

---

## Configurations Tested

| Config | RL Selection | DP | Fairness Reward |
|--------|--------------|-----|-----------------|
| **Full System** | ✓ | ✓ | ✓ |
| **No-RL** | ✗ (All clients) | ✓ | N/A |
| **No-DP** | ✓ | ✗ | ✓ |
| **No-Fairness** | ✓ | ✓ | ✗ |

---

## Metrics Tracked

- Final global accuracy
- Fairness gap (max - min client accuracy)
- Convergence speed
- Per-client accuracy variance

---

## Results Interpretation

### If Hypothesis is Proven ✓

**Paper claims:**
> "Ablation study confirms that each component contributes to the overall system. RL selection improves accuracy by X%, DP provides privacy with Y% cost, and the fairness reward reduces client accuracy variance by Z%. The full system achieves the best balance of accuracy, privacy, and fairness."

**Table format:**
| Configuration | Accuracy | Fairness Gap | ε |
|---------------|----------|--------------|---|
| Full System | X.XXXX | X.XXXX | X.XX |
| No-RL | -X.XX% | +X.XX% | X.XX |
| No-DP | +X.XX% | X.XXXX | ∞ |
| No-Fairness | X.XXXX | +X.XX% | X.XX |

### If Hypothesis is Not Proven ✗

**Alternative narrative:**
> "Ablation analysis reveals that [component X] has minimal impact on performance, suggesting the system can be simplified without significant degradation. However, [component Y] remains essential for [privacy/fairness]."

**Adjust claims based on results:**
- If No-RL ≈ Full: "FL is robust to client selection"
- If No-DP ≈ Full: "DP noise has minimal impact at current levels"
- If No-Fairness ≈ Full: "Fairness emerges naturally from the training process"

---

## Running

```bash
# Full experiment
python experiment.py --seeds 3 --rounds 20

# Quick test
python experiment.py --seeds 1 --rounds 5 --max-samples 50
```

---

## Output Files

- `figures/ablation_bar_chart.png` - Fig 3
- `tables/ablation_results.csv` - Table 3
