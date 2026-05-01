"""
Training loop for EG-BDQN / DQN / Random-Gating / Random-Oracle.

Usage:
    python train.py                                          # defaults from config.yaml
    python train.py --run_name my_experiment                  # custom run name
    python train.py --mode dqn --num_heads 1                  # DQN baseline
    python train.py --mode random_gating --B_total 5000       # random gating baseline
    python train.py --mode random_oracle --B_total 5000       # random oracle baseline

Observations are used as-is from the environment (dict with 'image' and 'mission').
"""

import argparse
import csv
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import yaml
from tqdm import tqdm

from agent import EGBDQNAgent
from model import tokenize_mission
from replay_buffer import DualReplayBuffer




def main():
    parser = argparse.ArgumentParser(description="EG-BDQN Training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run_name", default=None,
                        help="Descriptive name for this experiment run")
    parser.add_argument("--mode", default=None,
                        choices=["eg_bdqn", "dqn", "random_gating", "random_oracle"])
    parser.add_argument("--B_total", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--bootstrap_mask_prob", type=float, default=None)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.mode is not None:
        config["mode"] = args.mode
    if args.B_total is not None:
        config["budget"]["B_total"] = args.B_total
    if args.num_heads is not None:
        config["bdqn"]["num_heads"] = args.num_heads
    if args.total_steps is not None:
        config["training"]["total_steps"] = args.total_steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.bootstrap_mask_prob is not None:
        config["bdqn"]["bootstrap_mask_prob"] = args.bootstrap_mask_prob

    # ── Run name ──────────────────────────────────────────────────────────
    if args.run_name is None:
        mode = config["mode"]
        budget = config["budget"]["B_total"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{mode}_B{budget}_{timestamp}"
    else:
        run_name = args.run_name

    # ── Logging setup (nested: logs/<run_name>/) ──────────────────────────
    if args.log_dir:
        config["logging"]["log_dir"] = args.log_dir
    base_log_dir = config["logging"]["log_dir"]
    run_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "logs.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Mode: {config['mode']} | "
          f"Budget: {config['budget']['B_total']} | Run: {run_name}")
    print(f"Logging to: {run_dir}/")

    # ── Seeding ───────────────────────────────────────────────────────────
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Environment ───────────────────────────────────────────────────────
    env = gym.make(config["env"]["env_name"])
    env.action_space.seed(seed)
    num_actions = env.action_space.n

    # ── Agent & Buffer ────────────────────────────────────────────────────
    agent = EGBDQNAgent(config, num_actions, device)
    buffer = DualReplayBuffer(config["training"]["buffer_size"], config["budget"]["B_total"])

    total_steps        = config["training"]["total_steps"]
    batch_size         = config["training"]["batch_size"]
    train_start        = config["training"].get("train_start", 1000)
    target_update_freq = config["training"]["target_update_freq"]
    checkpoint_freq    = config["logging"]["checkpoint_freq"]
    log_freq           = config["logging"]["log_freq"]

    # ── Tracking ──────────────────────────────────────────────────────────
    episode_returns = deque(maxlen=100)
    episode_return = 0.0
    episode_num = 0
    last_loss = 0.0
    last_uncertainty = 0.0

    fieldnames = [
        "step", "episode", "episode_return", "rolling_mean_return",
        "oracle_queries_used", "budget_remaining", "td_loss",
        "uncertainty", "epsilon",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # ── Initial reset ─────────────────────────────────────────────────────
    obs, _info = env.reset(seed=seed)
    agent.reset_oracle(env)

    # ── Training loop ─────────────────────────────────────────────────────
    pbar = tqdm(range(1, total_steps + 1), desc="Training", unit="step")

    for step in pbar:
        steps_remaining = total_steps - step

        action, oracle_used, uncertainty, obs_mission = agent.select_action(
            obs, env, steps_remaining
        )
        last_uncertainty = uncertainty

        next_obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        episode_return += reward

        next_obs_mission = tokenize_mission(next_obs["mission"])
        buffer.push(
            obs["image"], obs_mission, action, reward,
            next_obs["image"], next_obs_mission, done,
            is_demo=oracle_used,
        )

        # Train
        if len(buffer) >= train_start:
            batch = buffer.sample(batch_size, device)
            last_loss = agent.train_step(batch)

        # Target network update
        if step % target_update_freq == 0:
            agent.update_target()

        # Episode boundary
        if done:
            episode_returns.append(episode_return)
            rolling = np.mean(episode_returns) if episode_returns else 0.0
            episode_num += 1

            # Update tqdm postfix with latest episode info
            pbar.set_postfix({
                "ep": episode_num,
                "ret": f"{episode_return:.2f}",
                "roll": f"{rolling:.2f}",
                "oracle": agent.oracle_queries_total,
                "budget": agent.B_remaining,
            })

            episode_return = 0.0
            obs, _info = env.reset()
            agent.reset_oracle(env)
        else:
            obs = next_obs

        # Periodic CSV log
        if step % log_freq == 0:
            rolling = np.mean(episode_returns) if episode_returns else 0.0
            writer.writerow({
                "step": step,
                "episode": episode_num,
                "episode_return": episode_returns[-1] if episode_returns else 0.0,
                "rolling_mean_return": round(rolling, 4),
                "oracle_queries_used": agent.oracle_queries_total,
                "budget_remaining": agent.B_remaining,
                "td_loss": round(last_loss, 6),
                "uncertainty": round(last_uncertainty, 6),
                "epsilon": round(agent.get_epsilon(), 4),
            })
            csv_file.flush()

        # Checkpoint
        if step % checkpoint_freq == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_{step}.pt")
            agent.save(ckpt_path)

    pbar.close()

    # ── Final save ────────────────────────────────────────────────────────
    agent.save(os.path.join(run_dir, "final.pt"))
    csv_file.close()
    env.close()
    print(f"\n✓ Training complete. Logs → {run_dir}/")


if __name__ == "__main__":
    main()
