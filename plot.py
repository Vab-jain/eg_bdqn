"""
Plotting utilities for EG-BDQN experiments.

Usage:
    # Compare multiple runs (training curves)
    python plot.py --runs logs/eg_bdqn_run logs/dqn_run logs/random_gating_run

    # Oracle usage over time
    python plot.py --runs logs/eg_bdqn_run --plot oracle_usage

    # Return vs budget (needs multiple EG-BDQN runs with different B_total)
    python plot.py --runs logs/eg_bdqn_B1000 logs/eg_bdqn_B5000 logs/eg_bdqn_B10000 --plot budget_comparison

    # All plots
    python plot.py --runs logs/eg_bdqn_run logs/dqn_run --plot all
"""

import argparse
import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _get_grouped_runs(run_dirs):
    """
    Given a list of paths (either direct run directories or experiment directories containing seed folders),
    returns a dictionary mapping label to a list of DataFrames.
    """
    grouped_data = collections.defaultdict(list)
    for path in run_dirs:
        # Check if this path itself has a logs.csv
        if os.path.exists(os.path.join(path, "logs.csv")):
            grouped_data[os.path.basename(path)].append(pd.read_csv(os.path.join(path, "logs.csv")))
        else:
            # Assume it's an experiment directory with seed folders
            for seed_dir in os.listdir(path):
                seed_path = os.path.join(path, seed_dir)
                if os.path.isdir(seed_path):
                    for algo_dir in os.listdir(seed_path):
                        algo_path = os.path.join(seed_path, algo_dir)
                        csv_path = os.path.join(algo_path, "logs.csv")
                        if os.path.isfile(csv_path):
                            grouped_data[algo_dir].append(pd.read_csv(csv_path))
    return grouped_data


def plot_training_curves(run_dirs, output_dir="plots"):
    """Training curve: rolling mean return vs steps for each run."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    grouped_data = _get_grouped_runs(run_dirs)

    for label, dfs in grouped_data.items():
        if not dfs:
            continue
        min_len = min(len(df["step"]) for df in dfs)
        steps = dfs[0]["step"].values[:min_len]

        returns = []
        for df in dfs:
            returns.append(df["rolling_mean_return"].values[:min_len])

        returns = np.array(returns)
        mean = np.mean(returns, axis=0)
        std = np.std(returns, axis=0)

        p = ax.plot(steps, mean, label=label, alpha=0.8)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=p[0].get_color())

    ax.set_xlabel("Steps")
    ax.set_ylabel("Rolling Mean Return (window=100)")
    ax.set_title("Training Curves")
    ax.legend(fontsize=9)
    out = os.path.join(output_dir, "training_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_oracle_usage(run_dirs, output_dir="plots"):
    """Cumulative oracle queries vs steps."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    grouped_data = _get_grouped_runs(run_dirs)

    for label, dfs in grouped_data.items():
        if not dfs or "oracle_queries_used" not in dfs[0].columns:
            continue
        min_len = min(len(df["step"]) for df in dfs)
        steps = dfs[0]["step"].values[:min_len]

        queries = []
        for df in dfs:
            if "oracle_queries_used" in df.columns:
                queries.append(df["oracle_queries_used"].values[:min_len])

        if not queries:
            continue

        queries = np.array(queries)
        mean = np.mean(queries, axis=0)
        std = np.std(queries, axis=0)

        p = ax.plot(steps, mean, label=label, alpha=0.8)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=p[0].get_color())

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Oracle Queries")
    ax.set_title("Oracle Usage Over Time")
    ax.legend(fontsize=9)
    out = os.path.join(output_dir, "oracle_usage.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_budget_comparison(run_dirs, output_dir="plots"):
    """Final rolling return vs total budget for each run."""
    os.makedirs(output_dir, exist_ok=True)
    budgets, returns = [], []

    grouped_data = _get_grouped_runs(run_dirs)

    for label, dfs in grouped_data.items():
        if not dfs:
            continue
        # Use label (algorithm/run name) to extract budget B
        try:
            b = int([p for p in label.split("_") if p.startswith("B")][0][1:])
        except (IndexError, ValueError):
            b = 0
            
        budgets.append(b)
        # Average the final rolling mean return across seeds
        final_returns = [df["rolling_mean_return"].iloc[-1] for df in dfs]
        returns.append(np.mean(final_returns))

    fig, ax = plt.subplots()
    ax.plot(budgets, returns, "o-", markersize=8)
    ax.set_xlabel("Total Oracle Budget (B_total)")
    ax.set_ylabel("Final Rolling Mean Return")
    ax.set_title("Return vs Oracle Budget")
    out = os.path.join(output_dir, "budget_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="EG-BDQN Plotting")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run directories (each containing logs.csv)")
    parser.add_argument("--plot", default="training",
                        choices=["training", "oracle_usage", "budget_comparison", "all"])
    parser.add_argument("--output_dir", default="plots")
    args = parser.parse_args()

    if args.plot in ("training", "all"):
        plot_training_curves(args.runs, args.output_dir)
    if args.plot in ("oracle_usage", "all"):
        plot_oracle_usage(args.runs, args.output_dir)
    if args.plot in ("budget_comparison", "all"):
        plot_budget_comparison(args.runs, args.output_dir)


if __name__ == "__main__":
    main()
