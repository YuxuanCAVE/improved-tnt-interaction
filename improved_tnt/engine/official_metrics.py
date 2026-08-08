from __future__ import annotations

import torch


INTERPRET_LATERAL_THRESHOLD_M = 1.0
INTERPRET_LOW_SPEED_MPS = 1.4
INTERPRET_HIGH_SPEED_MPS = 11.0
INTERPRET_MIN_LONGITUDINAL_THRESHOLD_M = 1.0
INTERPRET_MAX_LONGITUDINAL_THRESHOLD_M = 2.0


def _validate_metric_inputs(
    pred_m: torch.Tensor,
    target_m: torch.Tensor,
    final_yaw: torch.Tensor,
    final_speed: torch.Tensor,
) -> None:
    if pred_m.ndim != 4:
        raise ValueError(f"pred_m must be [B,K,T,2], got {tuple(pred_m.shape)}")
    if target_m.ndim != 3:
        raise ValueError(f"target_m must be [B,T,2], got {tuple(target_m.shape)}")
    if pred_m.shape[0] != target_m.shape[0] or pred_m.shape[2:] != target_m.shape[1:]:
        raise ValueError(
            "Prediction and target shapes are incompatible: "
            f"pred_m={tuple(pred_m.shape)}, target_m={tuple(target_m.shape)}"
        )
    if pred_m.shape[1] < 1 or pred_m.shape[-1] != 2:
        raise ValueError(f"pred_m must contain at least one 2D trajectory, got {tuple(pred_m.shape)}")
    expected_state_shape = (pred_m.shape[0],)
    if tuple(final_yaw.shape) != expected_state_shape or tuple(final_speed.shape) != expected_state_shape:
        raise ValueError(
            "final_yaw and final_speed must be [B]: "
            f"yaw={tuple(final_yaw.shape)}, speed={tuple(final_speed.shape)}, expected={expected_state_shape}"
        )
    for name, value in (
        ("pred_m", pred_m),
        ("target_m", target_m),
        ("final_yaw", final_yaw),
        ("final_speed", final_speed),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf values.")


def official_minade_minfde_mr(
    pred_m: torch.Tensor,
    target_m: torch.Tensor,
    final_yaw: torch.Tensor,
    final_speed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ICCV21 INTERPRET single-agent metrics for each case.

    pred_m can be shaped [B, T, 2] for one modality or [B, K, T, 2] for K
    modalities. target_m is [B, T, 2]. For the official challenge result,
    T=30 and at most six modalities should be supplied.
    """
    if pred_m.ndim == 3:
        pred_m = pred_m.unsqueeze(1)
    _validate_metric_inputs(pred_m, target_m, final_yaw, final_speed)

    dist = torch.linalg.norm(pred_m - target_m.unsqueeze(1), dim=-1)
    minade = dist.mean(dim=-1).min(dim=1).values
    minfde = dist[:, :, -1].min(dim=1).values

    final_delta = pred_m[:, :, -1] - target_m[:, None, -1]
    c = torch.cos(final_yaw).unsqueeze(1)
    s = torch.sin(final_yaw).unsqueeze(1)
    longitudinal_error = final_delta[..., 0] * c + final_delta[..., 1] * s
    lateral_error = -final_delta[..., 0] * s + final_delta[..., 1] * c

    speed = final_speed.clamp_min(0.0)
    longitudinal_threshold = torch.where(
        speed < INTERPRET_LOW_SPEED_MPS,
        torch.full_like(speed, INTERPRET_MIN_LONGITUDINAL_THRESHOLD_M),
        torch.where(
            speed > INTERPRET_HIGH_SPEED_MPS,
            torch.full_like(speed, INTERPRET_MAX_LONGITUDINAL_THRESHOLD_M),
            INTERPRET_MIN_LONGITUDINAL_THRESHOLD_M
            + (speed - INTERPRET_LOW_SPEED_MPS)
            / (INTERPRET_HIGH_SPEED_MPS - INTERPRET_LOW_SPEED_MPS),
        ),
    ).unsqueeze(1)
    modality_miss = (lateral_error.abs() > INTERPRET_LATERAL_THRESHOLD_M) | (
        longitudinal_error.abs() > longitudinal_threshold
    )
    case_miss = modality_miss.all(dim=1).float()
    return minade, minfde, case_miss
