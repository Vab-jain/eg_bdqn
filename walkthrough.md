# EG-BDQN Implementation Walkthrough

## What Was Built

A minimal research prototype for **Epistemic-Gated Bootstrapped DQN** on `BabyAI-GoToDoor-v0`. The agent uses ensemble Q-head disagreement to estimate epistemic uncertainty and selectively queries a BabyAI Bot oracle under a finite budget.

## File Structure

```
eg_bdqn/
├── config.yaml          # All hyperparameters
├── model.py             # BDQN: CNN encoder + mission encoder + N Q-heads
├── replay_buffer.py     # Simple circular replay buffer
├── agent.py             # Action selection, uncertainty, oracle gating (4 modes)
├── train.py             # Training loop with tqdm, CLI, nested logging
├── plot.py              # Plotting utilities
├── requirements.txt     # Dependencies
└── logs/                # Created at runtime
    └── <run_name>/      # One folder per experiment
        ├── logs.csv     # Training metrics
        ├── checkpoint_*.pt
        └── final.pt
```

## Architecture

```
Image (7×7×3) ──→ CNN (Conv2d×3, 2×2 kernels) ──→ FC ──→ 128-dim ─┐
                                                            ├→ Fusion(160→128) ──→ N Q-heads
Mission (str) ──→ Hash tokenize → Embed → Pool ──────────→ 32-dim ─┘
```

- **CNN**: 3 conv layers (3→16→32→64, kernel=2×2, no padding) — **MiniGrid recommended architecture** from the [training guide](https://minigrid.farama.org/content/training/)
- **Observation**: Uses default MiniGrid dict observation (`obs['image']` and `obs['mission']`)
- **Mission**: Hash-based bag-of-words embedding (64-bucket vocab, 32-dim), mean-pooled
- **Q-heads**: N independent `Linear(128, 7)` layers

## Changes Made

### 1. Recommended CNN Architecture

Upgraded to the official MiniGrid observation CNN pipeline:
- Retained the default MiniGrid dict observation (`obs['image']` and `obs['mission']`)
- CNN architecture from the [training guide](https://minigrid.farama.org/content/training/): `Conv2d(3→16, 2×2)` → `Conv2d(16→32, 2×2)` → `Conv2d(32→64, 2×2)` with dynamic output dim computation

### 2. Random Oracle Baseline

New mode `"random_oracle"`:
- Uses the **same uncertainty gating** as `eg_bdqn` (same `tau_t` threshold)
- But returns a **random action** instead of the BabyAI Bot action
- Tests whether the value comes from *when* to query vs *what* the oracle says

### 3. tqdm Progress Bar

Training loop now shows:
```
Training:  75%|▋| 1491/2000 [00:43<00:26, 18.91step/s, ep=6, ret=0.66, roll=0.52, oracle=20, budget=80]
```

### 4. Nested Log Directories

Logs are now saved as: `logs/<run_name>/logs.csv` and `logs/<run_name>/*.pt`

## 4 Modes

| Mode | Oracle | Gating | Purpose |
|---|---|---|---|
| `eg_bdqn` | BabyAI Bot | Uncertainty-based (`U_ep > tau_t`) | **Main method** |
| `dqn` | None | — | No-oracle baseline |
| `random_gating` | BabyAI Bot | Random (probability-based) | Tests value of *when* to query |
| `random_oracle` | Random action | Uncertainty-based (`U_ep > tau_t`) | Tests value of *what* oracle says |

## Sanity Check Results

| Mode | Steps | Oracle Queries | Budget | Status |
|---|---|---|---|---|
| `eg_bdqn` | 2000 | 20+ | 100→80 | ✅ |
| `dqn` | 500 | 0 | — | ✅ |
| `random_oracle` | 2000 | 20+ | 100→80 | ✅ |

---

## How to Run

```bash
# EG-BDQN (main method)
python train.py --run_name my_egbdqn

# DQN baseline
python train.py --mode dqn --num_heads 1 --run_name baseline_dqn

# Random gating (real oracle, random timing)
python train.py --mode random_gating --run_name baseline_random_gating

# Random oracle (random actions, uncertainty timing)
python train.py --mode random_oracle --run_name baseline_random_oracle

# Different budgets
python train.py --B_total 1000  --run_name eg_bdqn_B1000
python train.py --B_total 5000  --run_name eg_bdqn_B5000
python train.py --B_total 10000 --run_name eg_bdqn_B10000

# With bootstrap masking
python train.py --bootstrap_mask_prob 0.5 --run_name eg_bdqn_masked
```

## Plotting

```bash
# Training curves comparison
python plot.py --runs logs/my_egbdqn logs/baseline_dqn logs/baseline_random_gating --plot training

# Oracle usage
python plot.py --runs logs/my_egbdqn logs/baseline_random_gating --plot oracle_usage

# Budget comparison
python plot.py --runs logs/eg_bdqn_B1000 logs/eg_bdqn_B5000 logs/eg_bdqn_B10000 --plot budget_comparison

# All plots
python plot.py --runs logs/my_egbdqn logs/baseline_dqn --plot all
```
