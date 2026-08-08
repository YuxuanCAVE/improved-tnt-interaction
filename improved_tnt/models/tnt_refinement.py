from __future__ import annotations

import torch
from torch import nn


class TNTWeightedTrajectoryRefiner(nn.Module):
    """Refine a score-weighted TNT trajectory with frozen scene context."""

    def __init__(
        self,
        graph_dim: int,
        future_steps: int,
        hidden_dim: int = 128,
        trajectory_hidden_dim: int = 64,
        dropout: float = 0.1,
        residual_limit: float = 0.02,
    ) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.residual_limit = float(residual_limit)
        self.trajectory_encoder = nn.GRU(
            input_size=2,
            hidden_size=trajectory_hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(graph_dim + trajectory_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.future_steps * 2),
        )
        final_layer = self.decoder[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        target_feature: torch.Tensor,
        weighted_trajectory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weighted_trajectory.ndim != 3 or weighted_trajectory.shape[1:] != (self.future_steps, 2):
            raise ValueError(
                f"weighted_trajectory must be [B,{self.future_steps},2], "
                f"got {tuple(weighted_trajectory.shape)}"
            )
        _, hidden = self.trajectory_encoder(weighted_trajectory)
        fused = torch.cat([target_feature, hidden[-1]], dim=-1)
        residual = self.decoder(fused).view(-1, self.future_steps, 2)
        residual = torch.tanh(residual) * self.residual_limit
        return weighted_trajectory + residual, residual
