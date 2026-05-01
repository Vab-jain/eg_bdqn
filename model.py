"""
Bootstrapped DQN network: shared CNN encoder + mission text encoder + N Q-heads.

CNN architecture follows the MiniGrid training guide:
  https://minigrid.farama.org/content/training/
  (Lucas Willems' rl-starter-files architecture)

ImgObsWrapper produces (7, 7, 3) observations.  We permute to (3, 7, 7)
channels-first before feeding into the CNN.
"""

import torch
import torch.nn as nn


# ─── Mission tokenization ────────────────────────────────────────────────────

MISSION_VOCAB_SIZE = 64   # hash-bucket vocabulary
MISSION_MAX_LEN = 12      # max words kept from mission string
MISSION_EMBED_DIM = 32    # embedding dimension per word


def tokenize_mission(mission: str) -> list[int]:
    """Hash-based bag-of-words tokenizer.

    Maps each word to an index in [1, VOCAB_SIZE-1] via hashing.
    Index 0 is reserved for padding.
    """
    words = mission.lower().split()
    indices = [(hash(w) % (MISSION_VOCAB_SIZE - 1)) + 1 for w in words[:MISSION_MAX_LEN]]
    indices += [0] * (MISSION_MAX_LEN - len(indices))
    return indices


# ─── Network ─────────────────────────────────────────────────────────────────

class BDQN(nn.Module):
    """Bootstrapped DQN with shared CNN + mission encoder and N independent Q-heads.

    CNN architecture from the MiniGrid training guide (2×2 kernels, no padding)
    which is proven to work well with the compact 7×7×3 grid observations.
    """

    def __init__(self, num_actions: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

        # --- Image encoder (MiniGrid recommended architecture) ---
        # Input: (B, 3, 7, 7) after permute
        # Conv layers use 2×2 kernels (no padding) → progressively smaller spatial dims
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2),   # → (B, 16, 6, 6)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),  # → (B, 32, 5, 5)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),  # → (B, 64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                      # → (B, 64*4*4 = 1024)
        )

        # Compute CNN output dim dynamically (in case obs shape changes)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 7, 7)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(),
        )

        # --- Mission encoder (bag-of-words embedding) ---
        self.mission_embed = nn.Embedding(
            MISSION_VOCAB_SIZE, MISSION_EMBED_DIM, padding_idx=0
        )

        # --- Fusion: concat(cnn_features, mission_features) → shared repr ---
        self.fusion = nn.Sequential(
            nn.Linear(128 + MISSION_EMBED_DIM, 128),
            nn.ReLU(),
        )

        # --- N independent Q-heads (Deepened to prevent Shared Representation Collapse) ---
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_actions)
                )
                for _ in range(num_heads)
            ]
        )

    # ──────────────────────────────────────────────────────────────────────

    def _encode_mission(self, mission_tokens: torch.Tensor) -> torch.Tensor:
        """Mean-pool word embeddings (ignoring padding)."""
        embedded = self.mission_embed(mission_tokens)            # (B, L, D)
        mask = (mission_tokens != 0).float().unsqueeze(-1)       # (B, L, 1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled                                            # (B, D)

    def forward(
        self, image: torch.Tensor, mission_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image:          (B, 7, 7, 3)  uint8 or float  (NHWC from ImgObsWrapper)
            mission_tokens: (B, L)        long

        Returns:
            Q-values:       (B, N, A)
        """
        # Image: NHWC → NCHW, normalize to ~[0,1]
        x = image.float() / 10.0
        x = x.permute(0, 3, 1, 2)                               # (B, 3, 7, 7)

        cnn_feat = self.cnn_fc(self.cnn(x))                      # (B, 128)
        mission_feat = self._encode_mission(mission_tokens)      # (B, 32)

        shared = self.fusion(torch.cat([cnn_feat, mission_feat], dim=1))  # (B, 128)

        q_values = torch.stack(
            [head(shared) for head in self.heads], dim=1
        )                                                        # (B, N, A)
        return q_values
