"""
EG-BDQN Agent: action selection, epistemic uncertainty, budget-aware oracle gating.

Modes (config["mode"]):
  - "eg_bdqn"        : uncertainty-gated oracle queries (main method)
  - "dqn"            : standard DQN, no oracle, single head
  - "random_gating"  : real oracle queried at random (same budget)
  - "random_oracle"  : oracle gives random actions (tests value of good advice)
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from model import BDQN, tokenize_mission
from minigrid.utils.baby_ai_bot import BabyAIBot


class EGBDQNAgent:
    def __init__(self, config, num_actions, device="cpu"):
        self.config = config
        self.device = device
        self.num_actions = num_actions
        self.mode = config["mode"]

        # Networks
        self.num_heads = config["bdqn"]["num_heads"] if self.mode != "dqn" else 1
        self.online_net = BDQN(num_actions, self.num_heads).to(device)
        self.target_net = BDQN(num_actions, self.num_heads).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config["training"]["learning_rate"]
        )

        # Budget & Uncertainty Thresholding
        self.B_total = config["budget"]["B_total"]
        self.B_remaining = self.B_total
        self.u_buffer = deque(maxlen=config["budget"].get("u_buffer_size", 1000))
        self.top_x_percent = config["budget"].get("top_x_percent", 10)
        self.eps_stability = config["budget"].get("epsilon", 1e-6)

        # Exploration
        self.eps_start = config["exploration"]["eps_start"]
        self.eps_end = config["exploration"]["eps_end"]
        self.eps_decay_steps = config["exploration"]["eps_decay_steps"]

        # Training
        self.gamma = config["training"]["gamma"]
        self.bootstrap_mask_prob = config["bdqn"].get("bootstrap_mask_prob", 1.0)

        # Oracle state
        self.oracle_queries_total = 0
        self.step_count = 0

    def get_epsilon(self):
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def reset_oracle(self, env):
        pass  # Bot is recreated per query

    def compute_uncertainty(self, obs_image, obs_mission):
        with torch.no_grad():
            img = torch.tensor(obs_image, dtype=torch.uint8, device=self.device).unsqueeze(0)
            mis = torch.tensor(obs_mission, dtype=torch.long, device=self.device).unsqueeze(0)
            Q_all = self.online_net(img, mis)  # (1, N, A)
            mean_Q = Q_all.mean(dim=1)         # (1, A)
            a_greedy = mean_Q.argmax(dim=1).item()

            if self.num_heads <= 1:
                return a_greedy, 0.0

            Q_at_a = Q_all[0, :, a_greedy]     # (N,)
            std_Q = Q_at_a.std().item()
            mean_Q_at_a = Q_at_a.mean().item()
            U_ep = std_Q / (abs(mean_Q_at_a) + self.eps_stability)
        return a_greedy, U_ep

    def select_action(self, obs, env, steps_remaining=1):
        """Full action-selection pipeline.

        Args:
            obs:             dict with 'image' (7,7,3) and 'mission' (str)
            env:             the gymnasium env (needed for BabyAI oracle)
            steps_remaining: remaining training steps (for random_gating budget pacing)

        Returns:
            (action, oracle_used, U_ep, obs_mission_tokens)
        """
        obs_image = obs["image"]
        obs_mission = tokenize_mission(obs["mission"])
        self.step_count += 1
        eps = self.get_epsilon()

        if np.random.random() < eps:
            action = np.random.randint(0, self.num_actions)
            return action, False, 0.0, obs_mission

        a_greedy, U_ep = self.compute_uncertainty(obs_image, obs_mission)
        action = a_greedy
        oracle_used = False

        # Add to rolling uncertainty buffer
        if self.mode in ("eg_bdqn", "random_oracle"):
            self.u_buffer.append(U_ep)

        # --- PREVIOUS LOGIC (commented out) ---
        # # Dynamic tau_t calculation (top X% of recent uncertainties)
        # if len(self.u_buffer) > 0:
        #     fraction_budget_remaining = max(0.0, self.B_remaining / self.B_total)
        #     dynamic_top_x_percent = self.top_x_percent * fraction_budget_remaining
        #     tau_t = np.percentile(self.u_buffer, 100 - dynamic_top_x_percent) if dynamic_top_x_percent > 0 else float('inf')
        # else:
        #     tau_t = 0.0
        # --------------------------------------

        # New: simple percentile threshold
        if len(self.u_buffer) > 0:
            tau_t = np.percentile(self.u_buffer, 100 - self.top_x_percent) if self.top_x_percent > 0 else float('inf')
        else:
            tau_t = 0.0

        gate_open = U_ep > tau_t
        fraction_budget_remaining = max(0.0, self.B_remaining / self.B_total)
        # At 100% budget, 5% random query chance. At 0% budget, 0% chance.
        random_query_prob = 0.05 * fraction_budget_remaining 

        should_query = gate_open or (np.random.rand() < random_query_prob)

        if self.mode == "eg_bdqn" and self.B_remaining > 0:
            if should_query:
                action = self._query_oracle(env)
                oracle_used = True

        elif self.mode == "random_gating" and self.B_remaining > 0:
            # Real oracle, random timing — match expected budget usage
            p = self.B_remaining / max(steps_remaining, 1)
            if np.random.random() < p:
                action = self._query_oracle(env)
                oracle_used = True

        elif self.mode == "random_oracle" and self.B_remaining > 0:
            # Random action instead of real oracle, same gating as eg_bdqn
            if should_query:
                action = self._query_random_oracle()
                oracle_used = True

        # mode == "dqn": never query oracle

        return action, oracle_used, U_ep, obs_mission

    def _query_oracle(self, env):
        """Query BabyAI bot for the optimal action."""
        bot = BabyAIBot(env)
        action = bot.replan()
        self.B_remaining -= 1
        self.oracle_queries_total += 1
        return action

    def _query_random_oracle(self):
        """Return a random action (bad oracle baseline)."""
        action = np.random.randint(0, self.num_actions)
        self.B_remaining -= 1
        self.oracle_queries_total += 1
        return action

    def train_step(self, batch):
        obs_img = batch["obs_image"]
        obs_mis = batch["obs_mission"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_obs_img = batch["next_obs_image"]
        next_obs_mis = batch["next_obs_mission"]
        dones = batch["done"]
        batch_size = obs_img.shape[0]

        Q_all = self.online_net(obs_img, obs_mis)
        act_idx = actions.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, 1)
        Q_selected = Q_all.gather(2, act_idx).squeeze(2)

        with torch.no_grad():
            Q_t_all = self.target_net(next_obs_img, next_obs_mis)
            Q_t_mean = Q_t_all.mean(dim=1)
            max_Q_t = Q_t_mean.max(dim=1)[0]
            td_target = rewards + self.gamma * max_Q_t * (1.0 - dones)
            td_target = td_target.unsqueeze(1).expand(-1, self.num_heads)

        per_head_loss = (Q_selected - td_target).pow(2)

        if self.bootstrap_mask_prob < 1.0:
            mask = torch.bernoulli(
                torch.full((batch_size, self.num_heads), self.bootstrap_mask_prob, device=self.device)
            )
            loss = (mask * per_head_loss).sum() / mask.sum().clamp(min=1)
        else:
            loss = per_head_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "B_remaining": self.B_remaining,
            "oracle_queries_total": self.oracle_queries_total,
            "u_buffer": list(self.u_buffer),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
        self.B_remaining = ckpt["B_remaining"]
        self.oracle_queries_total = ckpt["oracle_queries_total"]
        if "u_buffer" in ckpt:
            self.u_buffer = deque(ckpt["u_buffer"], maxlen=self.u_buffer.maxlen)
