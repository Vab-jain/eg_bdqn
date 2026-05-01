# EG-BDQN

A minimal research prototype for **Epistemic-Gated Bootstrapped DQN** on `BabyAI-GoToDoor-v0`. The agent uses ensemble Q-head disagreement to estimate epistemic uncertainty and selectively queries a BabyAI Bot oracle under a finite budget.

*Note: This project is currently in active development and subject to significant changes.*

## Reports

- [Full Technical Report](https://github.com/Vab-jain/eg_bdqn/blob/main/report/preliminary_findings/preliminary_findings.pdf)
- [Short Research Summary](https://github.com/Vab-jain/eg_bdqn/blob/main/report/preliminary_summary/executive_summary.pdf)

## Execution Modes

| Mode | Oracle | Gating | Purpose |
|---|---|---|---|
| `eg_bdqn` | BabyAI Bot | Uncertainty-based (`U_ep > tau_t`) | **Main method** |
| `dqn` | None | — | No-oracle baseline |
| `random_gating` | BabyAI Bot | Random (probability-based) | Tests value of *when* to query |
| `random_oracle` | Random action | Uncertainty-based (`U_ep > tau_t`) | Tests value of *what* oracle says |

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

# With bootstrap masking
python train.py --bootstrap_mask_prob 0.5 --run_name eg_bdqn_masked
```

## Plotting

```bash
# Training curves comparison
python plot.py --runs logs/my_egbdqn logs/baseline_dqn logs/baseline_random_gating --plot training

# Oracle usage
python plot.py --runs logs/my_egbdqn logs/baseline_random_gating --plot oracle_usage

# All plots
python plot.py --runs logs/my_egbdqn logs/baseline_dqn --plot all
```
