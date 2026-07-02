from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from improved_tnt.data.common import CoordinateNormalizer
from improved_tnt.utils.geometry import local_to_global_tensor
from improved_tnt.utils.io import to_device


def _official_miss_rate(
    pred_m: torch.Tensor,
    target_m: torch.Tensor,
    anchor: torch.Tensor | None,
    fallback_threshold_m: float,
) -> torch.Tensor:
    """INTERACTION challenge miss rule for unimodal predictions.

    Requires anchor columns:
    [anchor_x, anchor_y, anchor_yaw, gt_final_yaw, gt_final_speed].
    Older caches only contain the first three columns, so they fall back to the
    previous fixed-FDE threshold until rebuilt.
    """
    fde = torch.linalg.norm(pred_m[:, -1] - target_m[:, -1], dim=-1)
    if anchor is None or anchor.shape[-1] < 5:
        return (fde > fallback_threshold_m).float()

    final_delta = pred_m[:, -1] - target_m[:, -1]
    final_yaw = anchor[:, 3]
    final_speed = anchor[:, 4].clamp_min(0.0)

    c = torch.cos(final_yaw)
    s = torch.sin(final_yaw)
    longitudinal_error = final_delta[:, 0] * c + final_delta[:, 1] * s
    lateral_error = -final_delta[:, 0] * s + final_delta[:, 1] * c

    longitudinal_threshold = torch.where(
        final_speed < 1.4,
        torch.ones_like(final_speed),
        torch.where(
            final_speed > 11.0,
            torch.full_like(final_speed, 2.0),
            1.0 + (final_speed - 1.4) / (11.0 - 1.4),
        ),
    )
    return ((lateral_error.abs() > 1.0) | (longitudinal_error.abs() > longitudinal_threshold)).float()


def _prediction_tensor(pred):
    return pred[0] if isinstance(pred, tuple) else pred


def _trajectory_loss(pred: torch.Tensor, target: torch.Tensor, loss_fn: nn.Module) -> float:
    if pred.ndim == 3:
        return float(loss_fn(pred, target).item())
    if pred.ndim != 4:
        raise ValueError(f"Expected prediction [B,T,2] or [B,K,T,2], got {tuple(pred.shape)}")
    per_element = torch.nn.functional.smooth_l1_loss(pred, target.unsqueeze(1).expand_as(pred), reduction="none")
    per_mode = per_element.mean(dim=(2, 3))
    return float(per_mode.min(dim=1).values.mean().item())


def _official_miss_rate_multimodal(
    pred_m: torch.Tensor,
    target_m: torch.Tensor,
    anchor: torch.Tensor | None,
    fallback_threshold_m: float,
) -> torch.Tensor:
    if pred_m.ndim == 3:
        return _official_miss_rate(pred_m, target_m, anchor, fallback_threshold_m)
    if pred_m.ndim != 4:
        raise ValueError(f"Expected prediction [B,T,2] or [B,K,T,2], got {tuple(pred_m.shape)}")
    if anchor is None or anchor.shape[-1] < 5:
        fde = torch.linalg.norm(pred_m[:, :, -1] - target_m[:, None, -1], dim=-1)
        return (fde > fallback_threshold_m).all(dim=1).float()

    final_delta = pred_m[:, :, -1] - target_m[:, None, -1]
    final_yaw = anchor[:, 3]
    final_speed = anchor[:, 4].clamp_min(0.0)
    c = torch.cos(final_yaw).unsqueeze(1)
    s = torch.sin(final_yaw).unsqueeze(1)
    longitudinal_error = final_delta[..., 0] * c + final_delta[..., 1] * s
    lateral_error = -final_delta[..., 0] * s + final_delta[..., 1] * c
    longitudinal_threshold = torch.where(
        final_speed < 1.4,
        torch.ones_like(final_speed),
        torch.where(
            final_speed > 11.0,
            torch.full_like(final_speed, 2.0),
            1.0 + (final_speed - 1.4) / (11.0 - 1.4),
        ),
    ).unsqueeze(1)
    miss_by_mode = (lateral_error.abs() > 1.0) | (longitudinal_error.abs() > longitudinal_threshold)
    return miss_by_mode.all(dim=1).float()


def evaluate_sequence(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    normalizer: CoordinateNormalizer,
    sensor_range_m: float | None = None,
    miss_threshold_m: float = 2.0,
) -> tuple[float, float, float, float]:
    model.eval()
    losses: list[float] = []
    ade_values: list[torch.Tensor] = []
    fde_values: list[torch.Tensor] = []
    miss_values: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                x, y, anchor, map_tokens, map_token_mask = batch
                anchor = anchor.to(device, non_blocking=True)
                map_tokens = map_tokens.to(device, non_blocking=True)
                map_token_mask = map_token_mask.to(device, non_blocking=True)
                map_aux = (map_tokens, map_token_mask)
            elif len(batch) == 4:
                x, y, anchor, map_image = batch
                anchor = anchor.to(device, non_blocking=True)
                map_image = map_image.to(device, non_blocking=True)
                map_aux = map_image
            elif len(batch) == 3:
                x, y, anchor = batch
                anchor = anchor.to(device, non_blocking=True)
                map_aux = None
            else:
                x, y = batch
                anchor = None
                map_aux = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if isinstance(map_aux, tuple):
                pred = _prediction_tensor(model(x, map_aux[0], map_aux[1]))
            else:
                pred = _prediction_tensor(model(x, map_aux) if map_aux is not None else model(x))
            losses.append(float(loss_fn(pred, y).item()))
            if anchor is not None:
                if sensor_range_m is None:
                    raise ValueError("sensor_range_m is required for ego-local sequence evaluation.")
                pred_m = local_to_global_tensor(pred, anchor, sensor_range_m)
                y_m = local_to_global_tensor(y, anchor, sensor_range_m)
            else:
                pred_m = normalizer.denormalize_xy_tensor(pred)
                y_m = normalizer.denormalize_xy_tensor(y)
            if pred_m.ndim == 3:
                dist = torch.linalg.norm(pred_m - y_m, dim=-1)
                ade_values.append(dist.mean(dim=1).cpu())
                fde_values.append(dist[:, -1].cpu())
            else:
                dist = torch.linalg.norm(pred_m - y_m.unsqueeze(1), dim=-1)
                ade_values.append(dist.mean(dim=-1).min(dim=1).values.cpu())
                fde_values.append(dist[:, :, -1].min(dim=1).values.cpu())
            miss_values.append(_official_miss_rate_multimodal(pred_m, y_m, anchor, miss_threshold_m).cpu())

    return (
        float(np.mean(losses)),
        torch.cat(ade_values).mean().item(),
        torch.cat(fde_values).mean().item(),
        torch.cat(miss_values).mean().item(),
    )


def evaluate_polyline(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    coordinate_scale_m: float,
    miss_threshold_m: float = 2.0,
) -> tuple[float, float, float, float]:
    model.eval()
    losses: list[float] = []
    ade_values: list[torch.Tensor] = []
    fde_values: list[torch.Tensor] = []
    miss_values: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            if "target_candidates" in batch and "candidate_mask" in batch:
                pred = _prediction_tensor(
                    model(
                        batch["polylines"],
                        batch["polyline_mask"],
                        batch["segment_mask"],
                        batch["polyline_types"],
                        target_candidates=batch["target_candidates"],
                        candidate_mask=batch["candidate_mask"],
                    )
                )
            else:
                pred = _prediction_tensor(
                    model(batch["polylines"], batch["polyline_mask"], batch["segment_mask"], batch["polyline_types"])
                )
            target = batch["target"]
            losses.append(_trajectory_loss(pred, target, loss_fn))

            pred_m = local_to_global_tensor(pred, batch["anchor"], coordinate_scale_m)
            target_m = local_to_global_tensor(target, batch["anchor"], coordinate_scale_m)
            if pred_m.ndim == 3:
                dist = torch.linalg.norm(pred_m - target_m, dim=-1)
                ade_values.append(dist.mean(dim=1).cpu())
                fde_values.append(dist[:, -1].cpu())
            else:
                dist = torch.linalg.norm(pred_m - target_m.unsqueeze(1), dim=-1)
                ade_values.append(dist.mean(dim=-1).min(dim=1).values.cpu())
                fde_values.append(dist[:, :, -1].min(dim=1).values.cpu())
            miss_values.append(_official_miss_rate_multimodal(pred_m, target_m, batch["anchor"], miss_threshold_m).cpu())

    return (
        float(np.mean(losses)),
        torch.cat(ade_values).mean().item(),
        torch.cat(fde_values).mean().item(),
        torch.cat(miss_values).mean().item(),
    )


def evaluate_model(
    family: str,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    dataset: Any,
    miss_threshold_m: float = 2.0,
) -> tuple[float, float, float, float]:
    if family == "polyline":
        return evaluate_polyline(
            model,
            loader,
            loss_fn,
            device,
            dataset.coordinate_scale_m,
            miss_threshold_m=miss_threshold_m,
        )
    return evaluate_sequence(
        model,
        loader,
        loss_fn,
        device,
        dataset.normalizer,
        sensor_range_m=dataset.sensor_range_m
        if dataset.representation
        in {"spatial_grid", "agent_set", "agent_set_raster_map", "agent_set_map_tokens", "agent_scene_raster"}
        else None,
        miss_threshold_m=miss_threshold_m,
    )
