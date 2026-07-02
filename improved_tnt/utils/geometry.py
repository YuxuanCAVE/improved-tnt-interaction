from __future__ import annotations

import torch


def local_to_global_tensor(local_xy: torch.Tensor, anchor: torch.Tensor, scale_m: float) -> torch.Tensor:
    rel = local_xy * scale_m
    psi = anchor[:, 2]
    c = torch.cos(psi)
    s = torch.sin(psi)
    view_shape = (anchor.shape[0],) + (1,) * (rel.ndim - 2)
    c = c.view(view_shape)
    s = s.view(view_shape)
    anchor_x = anchor[:, 0].view(view_shape)
    anchor_y = anchor[:, 1].view(view_shape)
    x = rel[..., 0] * c - rel[..., 1] * s + anchor_x
    y = rel[..., 0] * s + rel[..., 1] * c + anchor_y
    return torch.stack([x, y], dim=-1)
