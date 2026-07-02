from __future__ import annotations

import argparse
import random
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from improved_tnt.data.common import scene_name_from_track_path
from improved_tnt.data.factory import build_training_dataset, print_training_dataset_summary
from improved_tnt.engine.evaluate import evaluate_model
from improved_tnt.losses import TNTLoss
from improved_tnt.models import build_model_for_dataset, model_family
from improved_tnt.utils.config import config_namespace
from improved_tnt.utils.io import (
    copy_config,
    create_run_dir,
    default_device,
    plot_training_curves,
    print_device,
    print_epoch_progress,
    timestamp,
    to_device,
    write_metrics_csv,
)
from improved_tnt.utils.seed import set_seed


DEFAULT_CONFIG = Path("configs/experiments/train/polyline/tnt_vectornet.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train any configured trajectory prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cli_args = parser.parse_args()
    args = config_namespace(cli_args.config)
    args.config = cli_args.config
    return args


def _loss_fn(args: argparse.Namespace, family: str) -> nn.Module:
    loss_name = str(getattr(args, "loss", "mse")).lower()
    if family == "polyline" and loss_name == "huber":
        return nn.SmoothL1Loss()
    if loss_name == "huber":
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def _temporal_weights(
    target: torch.Tensor,
    weighting: str,
    start: float = 0.5,
    end: float = 1.5,
) -> torch.Tensor | None:
    if weighting in {"", "none", "null"}:
        return None
    if weighting != "linear":
        raise ValueError("temporal_loss_weighting must be one of: null, none, linear")
    steps = target.shape[-2]
    weights = torch.linspace(start, end, steps, device=target.device, dtype=target.dtype)
    return weights / weights.mean()


def _trajectory_huber_loss(pred: torch.Tensor, target: torch.Tensor, temporal_weighting: str) -> torch.Tensor:
    per_element = F.smooth_l1_loss(pred, target, reduction="none")
    weights = _temporal_weights(target, temporal_weighting)
    if weights is not None:
        per_element = per_element * weights.view((1,) * (per_element.ndim - 2) + (target.shape[-2], 1))
    return per_element.mean()


def _trajectory_huber_loss_by_mode(
    pred: torch.Tensor,
    target: torch.Tensor,
    temporal_weighting: str,
) -> torch.Tensor:
    per_element = F.smooth_l1_loss(pred, target, reduction="none")
    weights = _temporal_weights(target, temporal_weighting)
    if weights is not None:
        per_element = per_element * weights.view(1, 1, target.shape[-2], 1)
    return per_element.mean(dim=(2, 3))


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0**2)


def _format_memory_mb(value: float) -> str:
    return f"{value:.1f} MB" if value > 0.0 else "n/a"


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def _train_sequence_batch(
    model: nn.Module,
    batch,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(batch) == 5:
        x, y, _anchor, map_tokens, map_token_mask = batch
        map_tokens = map_tokens.to(device, non_blocking=True)
        map_token_mask = map_token_mask.to(device, non_blocking=True)
        map_aux = (map_tokens, map_token_mask)
    elif len(batch) == 4:
        x, y, _anchor, map_image = batch
        map_image = map_image.to(device, non_blocking=True)
        map_aux = map_image
    elif len(batch) == 3:
        x, y, _anchor = batch
        map_aux = None
    else:
        x, y = batch
        map_aux = None
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if isinstance(map_aux, tuple):
        pred = model(x, map_aux[0], map_aux[1])
    else:
        pred = model(x, map_aux) if map_aux is not None else model(x)
    loss = loss_fn(pred, y)
    return loss, {}


def _polyline_prediction_loss(
    pred,
    target: torch.Tensor,
    loss_fn: nn.Module,
    endpoint_loss_fn: nn.Module,
    temporal_weighting: str,
    mode_score_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    if isinstance(pred, tuple):
        trajectories, mode_scores = pred
    else:
        trajectories, mode_scores = pred, None

    parts: dict[str, float] = {}
    if trajectories.ndim == 3:
        if str(temporal_weighting).lower() in {"", "none", "null"}:
            loss = loss_fn(trajectories, target)
        else:
            loss = _trajectory_huber_loss(trajectories, target, temporal_weighting)
        return loss, parts, trajectories

    if trajectories.ndim != 4:
        raise ValueError(f"Expected trajectory prediction [B,T,2] or [B,K,T,2], got {tuple(trajectories.shape)}")

    target_by_mode = target.unsqueeze(1).expand_as(trajectories)
    per_mode_loss = _trajectory_huber_loss_by_mode(trajectories, target_by_mode, temporal_weighting)
    best_mode = per_mode_loss.argmin(dim=1)
    batch_idx = torch.arange(trajectories.shape[0], device=trajectories.device)
    best_pred = trajectories[batch_idx, best_mode]
    loss = per_mode_loss[batch_idx, best_mode].mean()

    if mode_scores is not None and mode_score_loss_weight > 0.0:
        score_loss = F.cross_entropy(mode_scores, best_mode)
        loss = loss + mode_score_loss_weight * score_loss
        parts["mode_score_loss"] = float(score_loss.item())
    parts["best_mode"] = float(best_mode.float().mean().item())
    return loss, parts, best_pred


def _train_tnt_batch(
    model: nn.Module,
    batch: dict,
    args: argparse.Namespace,
    dataset_coordinate_scale_m: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if "target_candidates" not in batch or "candidate_mask" not in batch:
        raise KeyError("tnt_vectornet requires target_candidates and candidate_mask in the dataset/cache.")
    target = batch["target"]
    pred = model(
        batch["polylines"],
        batch["polyline_mask"],
        batch["segment_mask"],
        batch["polyline_types"],
        target_candidates=batch["target_candidates"],
        candidate_mask=batch["candidate_mask"],
        target_endpoint=target[:, -1],
        return_dict=True,
    )
    scale_m = float(
        dataset_coordinate_scale_m
        or getattr(args, "coordinate_scale_m", None)
        or getattr(args, "sensor_range_m", 1.0)
    )
    loss_fn = TNTLoss(
        target_loss_weight=float(getattr(args, "target_loss_weight", 0.1)),
        motion_loss_weight=float(getattr(args, "motion_loss_weight", 1.0)),
        score_loss_weight=float(getattr(args, "score_loss_weight", 0.1)),
        score_temperature=float(getattr(args, "score_temperature", 0.01)),
        predicted_motion_loss_weight=float(getattr(args, "predicted_motion_loss_weight", 0.0)),
        endpoint_consistency_loss_weight=float(getattr(args, "endpoint_consistency_loss_weight", 1.0)),
        coordinate_scale_m=scale_m,
        target_loss_type=str(getattr(args, "target_loss_type", "soft")),
        target_soft_label_sigma_m=float(getattr(args, "target_soft_label_sigma_m", 1.0)),
        target_soft_label_radius_m=float(getattr(args, "target_soft_label_radius_m", 3.0)),
    )
    return loss_fn(pred, target, batch["target_candidates"], batch["candidate_mask"])


def _train_polyline_batch(
    model: nn.Module,
    batch,
    args: argparse.Namespace,
    device: torch.device,
    loss_fn: nn.Module,
    aux_loss_fn: nn.Module,
    endpoint_loss_fn: nn.Module,
    dataset_coordinate_scale_m: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = to_device(batch, device)
    if str(getattr(args, "model_type", "vectornet")) == "tnt_vectornet":
        return _train_tnt_batch(model, batch, args, dataset_coordinate_scale_m)
    aux_loss_weight = float(getattr(args, "aux_loss_weight", 0.0))
    aux_mask_ratio = float(getattr(args, "aux_mask_ratio", 0.15))
    endpoint_loss_weight = float(getattr(args, "endpoint_loss_weight", 0.0))
    temporal_weighting = str(getattr(args, "temporal_loss_weighting", "none")).lower()
    mode_score_loss_weight = float(getattr(args, "mode_score_loss_weight", 0.1))
    endpoint_teacher_forcing = bool(getattr(args, "endpoint_teacher_forcing", False))
    use_aux_loss = aux_loss_weight > 0.0 and bool(getattr(args, "auxiliary_node_loss", False))
    use_endpoint_loss = (
        str(getattr(args, "model_type", "vectornet")) == "mini_tnt_lstm_vectornet"
        and endpoint_loss_weight > 0.0
    )
    parts: dict[str, float] = {}

    if use_endpoint_loss:
        if use_aux_loss:
            pred, aux, endpoint = model(
                batch["polylines"],
                batch["polyline_mask"],
                batch["segment_mask"],
                batch["polyline_types"],
                return_aux=True,
                aux_mask_ratio=aux_mask_ratio,
                return_endpoint=True,
                target_endpoint=batch["target"][:, -1],
                endpoint_teacher_forcing=endpoint_teacher_forcing,
            )
            aux_loss = aux_loss_fn(aux["pred"], aux["target"])
            parts["aux_loss"] = float(aux_loss.item())
        else:
            pred, endpoint = model(
                batch["polylines"],
                batch["polyline_mask"],
                batch["segment_mask"],
                batch["polyline_types"],
                return_endpoint=True,
                target_endpoint=batch["target"][:, -1],
                endpoint_teacher_forcing=endpoint_teacher_forcing,
            )
            aux_loss = None
        pred_loss, pred_parts, _best_pred = _polyline_prediction_loss(
            pred,
            batch["target"],
            loss_fn,
            endpoint_loss_fn,
            temporal_weighting,
            mode_score_loss_weight,
        )
        parts.update(pred_parts)
        endpoint_loss = endpoint_loss_fn(endpoint, batch["target"][:, -1])
        loss = pred_loss + endpoint_loss_weight * endpoint_loss
        if aux_loss is not None:
            loss = loss + aux_loss_weight * aux_loss
        parts["endpoint_loss"] = float(endpoint_loss.item())
        return loss, parts

    if use_aux_loss:
        pred, aux = model(
            batch["polylines"],
            batch["polyline_mask"],
            batch["segment_mask"],
            batch["polyline_types"],
            return_aux=True,
            aux_mask_ratio=aux_mask_ratio,
        )
        pred_loss, pred_parts, _best_pred = _polyline_prediction_loss(
            pred,
            batch["target"],
            loss_fn,
            endpoint_loss_fn,
            temporal_weighting,
            mode_score_loss_weight,
        )
        loss = pred_loss
        parts.update(pred_parts)
        aux_loss = aux_loss_fn(aux["pred"], aux["target"])
        loss = loss + aux_loss_weight * aux_loss
        parts["aux_loss"] = float(aux_loss.item())
        return loss, parts

    pred = model(batch["polylines"], batch["polyline_mask"], batch["segment_mask"], batch["polyline_types"])
    loss, pred_parts, _best_pred = _polyline_prediction_loss(
        pred,
        batch["target"],
        loss_fn,
        endpoint_loss_fn,
        temporal_weighting,
        mode_score_loss_weight,
    )
    parts.update(pred_parts)
    return loss, parts


def _checkpoint_payload(
    args: argparse.Namespace,
    family: str,
    dataset,
    model: nn.Module,
    epoch: int,
    row: dict[str, float],
    miss_threshold_m: float,
) -> dict:
    payload = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "model_family": family,
        "model_type": getattr(args, "model_type", "lstm"),
        "epoch": epoch,
        "miss_threshold_m": miss_threshold_m,
        **row,
    }
    if family == "polyline":
        payload.update(
            {
                "input_dim": dataset.feature_dim,
                "future_steps": dataset.future_steps,
                "coordinate_scale_m": dataset.coordinate_scale_m,
                "target_mode": dataset.target_mode,
                "include_velocity": bool(getattr(dataset, "include_velocity", False)),
                "architecture": getattr(args, "architecture", "current"),
                "auxiliary_node_loss": bool(getattr(args, "auxiliary_node_loss", False)),
                "aux_loss_weight": float(getattr(args, "aux_loss_weight", 0.0)),
                "aux_mask_ratio": float(getattr(args, "aux_mask_ratio", 0.15)),
                "endpoint_loss_weight": float(getattr(args, "endpoint_loss_weight", 0.0)),
                "endpoint_teacher_forcing": bool(getattr(args, "endpoint_teacher_forcing", False)),
                "max_target_candidates": int(getattr(dataset, "max_target_candidates", 0)),
                "target_candidate_spacing_m": float(getattr(dataset, "target_candidate_spacing_m", 0.0)),
                "target_candidate_nearest_fraction": float(
                    getattr(dataset, "target_candidate_nearest_fraction", 0.5)
                ),
                "target_candidate_source": str(getattr(dataset, "target_candidate_source", "all")),
                "target_candidate_type_ids": (
                    list(getattr(dataset, "target_candidate_type_ids", []))
                    if getattr(dataset, "target_candidate_type_ids", None) is not None
                    else None
                ),
                "target_candidate_lateral_offsets_m": list(
                    getattr(dataset, "target_candidate_lateral_offsets_m", (0.0,))
                ),
                "target_candidate_grid_spacing_m": float(
                    getattr(dataset, "target_candidate_grid_spacing_m", 2.0)
                ),
                "target_candidate_lane_quota": getattr(dataset, "target_candidate_lane_quota", None),
                "target_candidate_grid_quota": getattr(dataset, "target_candidate_grid_quota", None),
                "target_candidate_selection": str(
                    getattr(dataset, "target_candidate_selection", "nearest_plus_radial_angular_coverage")
                ),
                "target_candidate_reachable_dt_s": float(
                    getattr(dataset, "target_candidate_reachable_dt_s", 0.1)
                ),
                "target_candidate_reachable_primary_fraction": float(
                    getattr(dataset, "target_candidate_reachable_primary_fraction", 0.6)
                ),
                "tnt_top_m": int(getattr(args, "tnt_top_m", 50)),
                "tnt_output_k": int(getattr(args, "tnt_output_k", 1)),
                "target_hidden_dim": int(getattr(args, "target_hidden_dim", getattr(args, "endpoint_hidden_dim", 128))),
                "motion_hidden_dim": int(getattr(args, "motion_hidden_dim", getattr(args, "decoder_hidden_dim", 128))),
                "score_hidden_dim": int(getattr(args, "score_hidden_dim", getattr(args, "decoder_hidden_dim", 128))),
                "target_offset_limit": float(getattr(args, "target_offset_limit", 0.05)),
                "use_refined_targets": bool(getattr(args, "use_refined_targets", True)),
                "trajectory_nms_threshold_m": float(getattr(args, "trajectory_nms_threshold_m", 2.0)),
                "target_loss_weight": float(getattr(args, "target_loss_weight", 0.1)),
                "motion_loss_weight": float(getattr(args, "motion_loss_weight", 1.0)),
                "score_loss_weight": float(getattr(args, "score_loss_weight", 0.1)),
                "score_temperature": float(getattr(args, "score_temperature", 0.01)),
                "predicted_motion_loss_weight": float(getattr(args, "predicted_motion_loss_weight", 0.0)),
                "endpoint_consistency_loss_weight": float(
                    getattr(args, "endpoint_consistency_loss_weight", 1.0)
                ),
                "target_loss_type": str(getattr(args, "target_loss_type", "soft")),
                "target_soft_label_sigma_m": float(getattr(args, "target_soft_label_sigma_m", 1.0)),
                "target_soft_label_radius_m": float(getattr(args, "target_soft_label_radius_m", 3.0)),
                "trajectory_loss_weight": float(getattr(args, "trajectory_loss_weight", 1.0)),
                "trajectory_score_loss_weight": float(getattr(args, "trajectory_score_loss_weight", 0.1)),
                "trajectory_score_temperature": float(getattr(args, "trajectory_score_temperature", 0.01)),
                "target_cls_loss_weight": float(getattr(args, "target_cls_loss_weight", 0.1)),
                "target_offset_loss_weight": float(getattr(args, "target_offset_loss_weight", 0.1)),
                "target_teacher_forcing": bool(getattr(args, "target_teacher_forcing", True)),
            }
        )
    else:
        model_input_dim = (
            dataset.grid_feature_dim
            if dataset.representation
            in {"spatial_grid", "agent_set", "agent_set_raster_map", "agent_set_map_tokens", "agent_scene_raster"}
            else dataset.input_dim
        )
        payload.update(
            {
                "input_dim": model_input_dim,
                "target_dim": dataset.target_dim,
                "future_steps": dataset.future_steps,
                "representation": dataset.representation,
                "neighbor_selection": getattr(dataset, "neighbor_selection", "timestep_nearest"),
                "neighbor_filter": getattr(dataset, "neighbor_filter", "none"),
                "heading_filter_cos": float(getattr(dataset, "heading_filter_cos", 0.5)),
                "target_mode": "ego_local"
                if dataset.representation in {"spatial_grid", "agent_set", "agent_set_raster_map", "agent_set_map_tokens"}
                or dataset.representation == "agent_scene_raster"
                else "global_normalized",
                "normalizer": vars(dataset.normalizer),
            }
        )
        if dataset.representation in {"agent_set_raster_map", "agent_scene_raster"}:
            payload.update(
                {
                    "map_range_m": float(getattr(dataset, "map_range_m", dataset.sensor_range_m)),
                    "map_image_size": int(getattr(dataset, "map_image_size", 128)),
                    "map_channels": int(getattr(dataset, "map_channels", 3)),
                    "raster_range_m": float(getattr(dataset, "map_range_m", dataset.sensor_range_m)),
                    "raster_image_size": int(getattr(dataset, "map_image_size", 128)),
                    "raster_channels": int(getattr(dataset, "map_channels", 3)),
                    "map_line_width": int(getattr(dataset, "map_line_width", 1)),
                }
            )
        if dataset.representation == "agent_set_map_tokens":
            payload.update(
                {
                    "map_range_m": float(getattr(dataset, "map_range_m", dataset.sensor_range_m)),
                    "max_map_tokens": int(getattr(dataset, "max_map_tokens", 128)),
                    "map_token_dim": int(getattr(dataset, "map_token_dim", 13)),
                }
            )
    return payload


def _build_lr_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    scheduler_name = str(getattr(args, "lr_scheduler", "none") or "none").lower()
    if scheduler_name in {"", "none", "null"}:
        return None
    if scheduler_name in {"reduce_on_plateau", "plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(args, "lr_scheduler_factor", 0.5)),
            patience=int(getattr(args, "lr_scheduler_patience", 4)),
            threshold=float(getattr(args, "lr_scheduler_threshold", 1e-3)),
            threshold_mode="rel",
            cooldown=int(getattr(args, "lr_scheduler_cooldown", 0)),
            min_lr=float(getattr(args, "lr_scheduler_min_lr", 1e-5)),
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(getattr(args, "lr_scheduler_t_max", getattr(args, "epochs", 50))),
            eta_min=float(getattr(args, "lr_scheduler_min_lr", 1e-5)),
        )
    raise ValueError("lr_scheduler must be one of: none, reduce_on_plateau, cosine")


def _scheduler_metric(row: dict[str, float], monitor: str) -> float:
    monitor_name = str(monitor or "fde_m")
    if monitor_name not in row:
        raise KeyError(f"lr_scheduler_monitor={monitor_name!r} is not available in metrics row.")
    return float(row[monitor_name])


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _dataset_refs(dataset) -> torch.Tensor | None:
    cached_ref = getattr(dataset, "ref", None)
    if cached_ref is not None:
        return cached_ref.long()
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return None
    return torch.tensor([[ref.file_idx, ref.track_id, ref.end_index] for ref in samples], dtype=torch.long)


def _dataset_scene_names(dataset) -> list[str] | None:
    scene_names = getattr(dataset, "scene_names", None)
    if scene_names:
        return [str(name) for name in scene_names]
    files = getattr(dataset, "files", None)
    if files:
        return [scene_name_from_track_path(track_file.path) for track_file in files]
    meta = getattr(dataset, "meta", None)
    if meta and meta.get("scene_names"):
        return [str(name) for name in meta["scene_names"]]
    return None


def _build_group_split(dataset, args: argparse.Namespace) -> tuple[Subset, Subset, str]:
    split_strategy = str(getattr(args, "split_strategy", getattr(args, "data_split", "window"))).lower()
    if split_strategy in {"window", "random", "random_window"}:
        val_size = max(1, int(len(dataset) * float(args.val_ratio)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(int(getattr(args, "seed", 7))),
        )
        return train_ds, val_ds, "window"

    if split_strategy not in {"track", "track_id", "file", "scenario", "scenario_holdout"}:
        raise ValueError(
            f"Unknown split_strategy={split_strategy!r}; "
            "expected window, track, file, or scenario."
        )

    refs = _dataset_refs(dataset)
    if refs is None:
        raise ValueError(
            f"split_strategy={split_strategy!r} requires per-sample refs. "
            "Rebuild the cache with the current precompute_cache.py or train without cache."
        )
    if refs.shape[0] != len(dataset):
        raise ValueError(f"Dataset ref count mismatch: refs={refs.shape[0]} windows={len(dataset)}")

    scene_names = _dataset_scene_names(dataset)
    groups: dict[object, list[int]] = defaultdict(list)
    for index, (file_idx, track_id, _end_index) in enumerate(refs.tolist()):
        if split_strategy in {"track", "track_id"}:
            key = (int(file_idx), int(track_id))
        elif split_strategy == "file":
            key = int(file_idx)
        else:
            if not scene_names:
                raise ValueError(
                    "scenario split requires file scene names. "
                    "Rebuild the cache with the current precompute_cache.py or train without cache."
                )
            key = scene_names[int(file_idx)]
        groups[key].append(index)

    val_scenarios = getattr(args, "val_scenarios", None)
    if split_strategy in {"scenario", "scenario_holdout"} and val_scenarios:
        val_group_keys = {str(name) for name in val_scenarios}
        unknown = val_group_keys - {str(key) for key in groups}
        if unknown:
            raise ValueError(f"val_scenarios contains unknown scenario(s): {sorted(unknown)}")
    else:
        rng = random.Random(int(getattr(args, "seed", 7)))
        shuffled = list(groups)
        rng.shuffle(shuffled)
        target_val = max(1, int(len(dataset) * float(args.val_ratio)))
        val_group_keys = set()
        val_count = 0
        for key in shuffled:
            if val_count >= target_val and val_group_keys:
                break
            val_group_keys.add(key)
            val_count += len(groups[key])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for key, indices in groups.items():
        if key in val_group_keys or str(key) in val_group_keys:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not train_indices or not val_indices:
        raise ValueError(
            f"split_strategy={split_strategy!r} produced empty train or val split "
            f"(train={len(train_indices)}, val={len(val_indices)})."
        )

    return Subset(dataset, train_indices), Subset(dataset, val_indices), split_strategy


def _build_train_val_datasets(args: argparse.Namespace):
    val_data_split = getattr(args, "val_data_split", None)
    if val_data_split is None:
        dataset = build_training_dataset(args)
        train_ds, val_ds, split_strategy = _build_group_split(dataset, args)
        return dataset, dataset, train_ds, val_ds, split_strategy

    train_dataset = build_training_dataset(args)
    val_args = argparse.Namespace(**vars(args))
    val_args.data_split = val_data_split
    val_args.scenes = getattr(args, "val_scenes", getattr(args, "scenes", None))
    val_args.max_files = getattr(args, "val_max_files", getattr(args, "max_files", None))
    val_args.use_cache = bool(getattr(args, "val_use_cache", False))
    if val_args.use_cache:
        val_args.cache_path = getattr(args, "val_cache_path")
        val_args.map_cache_path = getattr(args, "val_map_cache_path", getattr(args, "map_cache_path", None))
        val_args.cache_paths = None
    val_args.max_samples = getattr(args, "val_max_samples", None)
    val_args.samples_per_scene_type = getattr(
        args, "val_samples_per_scene_type", getattr(args, "samples_per_scene_type", None)
    )
    val_args.samples_per_scenario = getattr(
        args, "val_samples_per_scenario", getattr(args, "samples_per_scenario", None)
    )
    val_dataset = build_training_dataset(val_args, normalizer=getattr(train_dataset, "normalizer", None))
    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))
    return (
        train_dataset,
        val_dataset,
        Subset(train_dataset, train_indices),
        Subset(val_dataset, val_indices),
        f"file_split:{getattr(args, 'data_split', None)}->{val_data_split}",
    )


def main() -> None:
    args = parse_args()
    family = model_family(str(getattr(args, "model_type", "lstm")))
    set_seed(int(getattr(args, "seed", 7)))

    default_root = Path("runs") / str(getattr(args, "model_type", "model")) / "train"
    map_tag = "map" if bool(getattr(args, "include_map", False)) else "agent_only"
    suffix = f"{getattr(args, 'model_type', 'model')}_{map_tag}" if family == "polyline" else str(getattr(args, "model_type", "model"))
    run_dir = create_run_dir(args, default_root, f"{timestamp()}_{suffix}")
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    copy_config(args.config, run_dir)
    print(f"Run directory: {run_dir}")

    dataset, eval_dataset, train_ds, val_ds, split_strategy = _build_train_val_datasets(args)
    print_training_dataset_summary(dataset, args)
    if eval_dataset is not dataset:
        print("Validation dataset:")
        eval_summary_args = argparse.Namespace(**{**vars(args), "use_cache": not hasattr(eval_dataset, "samples")})
        print_training_dataset_summary(eval_dataset, eval_summary_args)
    train_size = len(train_ds)
    val_size = len(val_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f"Dataset windows: total={len(dataset)} train={train_size} val={val_size}; "
        f"split={split_strategy}; train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    device = default_device()
    print_device(device)
    model = build_model_for_dataset(args, dataset).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )
    scheduler = _build_lr_scheduler(args, optimizer)
    use_ema = bool(getattr(args, "use_ema", False))
    ema = ExponentialMovingAverage(model, decay=float(getattr(args, "ema_decay", 0.999))) if use_ema else None
    ema_validation_start_epoch = int(getattr(args, "ema_validation_start_epoch", 1))
    loss_fn = _loss_fn(args, family)
    aux_loss_fn = nn.SmoothL1Loss()
    endpoint_loss_fn = nn.SmoothL1Loss()
    miss_threshold_m = float(getattr(args, "miss_threshold_m", 2.0))

    best_ade = float("inf")
    best_fde = float("inf")
    best_mr = float("inf")
    best_checkpoint_path = weights_dir / "best_model.pt"
    best_ade_checkpoint_path = weights_dir / "best_ade_model.pt"
    best_fde_checkpoint_path = weights_dir / "best_fde_model.pt"
    best_mr_checkpoint_path = weights_dir / "best_mr_model.pt"
    last_checkpoint_path = weights_dir / "last_model.pt"
    metrics: list[dict[str, float]] = []
    train_log_interval = max(1, int(getattr(args, "train_log_interval", 20)))

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        aux_losses: list[float] = []
        endpoint_losses: list[float] = []
        final_losses: list[float] = []
        mode_score_losses: list[float] = []
        target_cls_losses: list[float] = []
        target_offset_losses: list[float] = []
        target_candidate_recalls: list[float] = []
        predicted_motion_losses: list[float] = []
        endpoint_consistency_losses: list[float] = []
        num_batches = len(train_loader)
        _sync_if_cuda(device)
        _reset_peak_memory_if_cuda(device)
        epoch_start = time.perf_counter()
        train_start = epoch_start

        for batch_idx, batch in enumerate(train_loader, start=1):
            if family == "polyline":
                loss, parts = _train_polyline_batch(
                    model,
                    batch,
                    args,
                    device,
                    loss_fn,
                    aux_loss_fn,
                    endpoint_loss_fn,
                    getattr(dataset, "coordinate_scale_m", None),
                )
            else:
                loss, parts = _train_sequence_batch(model, batch, device, loss_fn)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            train_losses.append(float(loss.item()))
            if "aux_loss" in parts:
                aux_losses.append(parts["aux_loss"])
            if "endpoint_loss" in parts:
                endpoint_losses.append(parts["endpoint_loss"])
            if "final_loss" in parts:
                final_losses.append(parts["final_loss"])
            if "mode_score_loss" in parts:
                mode_score_losses.append(parts["mode_score_loss"])
            if "target_cls_loss" in parts:
                target_cls_losses.append(parts["target_cls_loss"])
            if "target_offset_loss" in parts:
                target_offset_losses.append(parts["target_offset_loss"])
            if "target_candidate_recall_m" in parts:
                target_candidate_recalls.append(parts["target_candidate_recall_m"])
            if "predicted_motion_loss" in parts:
                predicted_motion_losses.append(parts["predicted_motion_loss"])
            if "endpoint_consistency_loss" in parts:
                endpoint_consistency_losses.append(parts["endpoint_consistency_loss"])

            should_log = batch_idx == 1 or batch_idx == num_batches or batch_idx % train_log_interval == 0
            if should_log:
                elapsed = time.perf_counter() - epoch_start
                batches_per_sec = batch_idx / max(elapsed, 1e-9)
                remaining = (num_batches - batch_idx) / max(batches_per_sec, 1e-9)
                print_epoch_progress(
                    epoch,
                    int(args.epochs),
                    batch_idx,
                    num_batches,
                    float(np.mean(train_losses)),
                    elapsed,
                    remaining,
                )

        _sync_if_cuda(device)
        train_time_s = time.perf_counter() - train_start
        train_peak_gpu_memory_mb = _peak_memory_mb(device)
        train_samples_per_s = train_size / max(train_time_s, 1e-9)

        _reset_peak_memory_if_cuda(device)
        _sync_if_cuda(device)
        val_start = time.perf_counter()
        eval_model = ema.model if ema is not None and epoch >= ema_validation_start_epoch else model
        val_loss, ade, fde, mr = evaluate_model(
            family,
            eval_model,
            val_loader,
            loss_fn,
            device,
            eval_dataset,
            miss_threshold_m=miss_threshold_m,
        )
        _sync_if_cuda(device)
        val_time_s = time.perf_counter() - val_start
        val_peak_gpu_memory_mb = _peak_memory_mb(device)
        val_samples_per_s = val_size / max(val_time_s, 1e-9)
        train_loss = float(np.mean(train_losses))
        epoch_elapsed = train_time_s + val_time_s
        print_epoch_progress(
            epoch,
            int(args.epochs),
            num_batches,
            num_batches,
            train_loss,
            epoch_elapsed,
            0.0,
            val_loss=val_loss,
            ade=ade,
            fde=fde,
            mr=mr,
            end=True,
        )
        print(
            "compute: "
            f"train_time={train_time_s:.2f}s train_samples_per_s={train_samples_per_s:.2f} "
            f"train_peak_gpu_memory={_format_memory_mb(train_peak_gpu_memory_mb)}; "
            f"val_time={val_time_s:.2f}s val_samples_per_s={val_samples_per_s:.2f} "
            f"val_peak_gpu_memory={_format_memory_mb(val_peak_gpu_memory_mb)}"
        )

        row = {
            "epoch": epoch,
            "lr": _current_lr(optimizer),
            "ema_validation": float(eval_model is not model),
            "train_loss": train_loss,
            "aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "endpoint_loss": float(np.mean(endpoint_losses)) if endpoint_losses else 0.0,
            "final_loss": float(np.mean(final_losses)) if final_losses else 0.0,
            "mode_score_loss": float(np.mean(mode_score_losses)) if mode_score_losses else 0.0,
            "target_cls_loss": float(np.mean(target_cls_losses)) if target_cls_losses else 0.0,
            "target_offset_loss": float(np.mean(target_offset_losses)) if target_offset_losses else 0.0,
            "target_candidate_recall_m": float(np.mean(target_candidate_recalls)) if target_candidate_recalls else 0.0,
            "predicted_motion_loss": float(np.mean(predicted_motion_losses)) if predicted_motion_losses else 0.0,
            "endpoint_consistency_loss": float(np.mean(endpoint_consistency_losses)) if endpoint_consistency_losses else 0.0,
            "val_loss": val_loss,
            "ade_m": ade,
            "fde_m": fde,
            "mr": mr,
            "train_time_s": train_time_s,
            "val_time_s": val_time_s,
            "epoch_time_s": epoch_elapsed,
            "train_samples_per_s": train_samples_per_s,
            "val_samples_per_s": val_samples_per_s,
            "train_peak_gpu_memory_mb": train_peak_gpu_memory_mb,
            "val_peak_gpu_memory_mb": val_peak_gpu_memory_mb,
        }
        if family != "polyline":
            row.pop("aux_loss")
            row.pop("endpoint_loss")
            row.pop("final_loss")
            row.pop("mode_score_loss")
            row.pop("target_cls_loss")
            row.pop("target_offset_loss")
            row.pop("target_candidate_recall_m")
            row.pop("predicted_motion_loss")
            row.pop("endpoint_consistency_loss")
        metrics.append(row)
        write_metrics_csv(metrics, run_dir / "metrics.csv")

        checkpoint_model = eval_model
        checkpoint = _checkpoint_payload(args, family, dataset, checkpoint_model, epoch, row, miss_threshold_m)
        checkpoint["ema_applied"] = bool(eval_model is not model)
        checkpoint["ema_decay"] = float(getattr(args, "ema_decay", 0.999)) if ema is not None else None
        torch.save(checkpoint, last_checkpoint_path)
        if ade < best_ade:
            best_ade = ade
            torch.save(checkpoint, best_ade_checkpoint_path)
            torch.save(checkpoint, best_checkpoint_path)
        if fde < best_fde:
            best_fde = fde
            torch.save(checkpoint, best_fde_checkpoint_path)
        if mr < best_mr:
            best_mr = mr
            torch.save(checkpoint, best_mr_checkpoint_path)

        if scheduler is not None:
            old_lr = _current_lr(optimizer)
            scheduler_name = str(getattr(args, "lr_scheduler", "none") or "none").lower()
            if scheduler_name in {"reduce_on_plateau", "plateau"}:
                scheduler.step(_scheduler_metric(row, str(getattr(args, "lr_scheduler_monitor", "fde_m"))))
            else:
                scheduler.step()
            new_lr = _current_lr(optimizer)
            if new_lr != old_lr:
                print(f"lr_scheduler: lr changed {old_lr:.6g} -> {new_lr:.6g}")

    plot_training_curves(metrics, run_dir / "curves.png")
    total_train_time_s = sum(float(row["train_time_s"]) for row in metrics)
    total_val_time_s = sum(float(row["val_time_s"]) for row in metrics)
    total_epoch_time_s = sum(float(row["epoch_time_s"]) for row in metrics)
    avg_train_samples_per_s = float(np.mean([row["train_samples_per_s"] for row in metrics]))
    avg_val_samples_per_s = float(np.mean([row["val_samples_per_s"] for row in metrics]))
    peak_train_gpu_memory_mb = max(float(row["train_peak_gpu_memory_mb"]) for row in metrics)
    peak_val_gpu_memory_mb = max(float(row["val_peak_gpu_memory_mb"]) for row in metrics)
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_ade_m: {best_ade:.6f}\n")
        f.write(f"best_fde_m: {best_fde:.6f}\n")
        f.write(f"best_mr: {best_mr:.6f}\n")
        f.write(f"use_ema: {bool(getattr(args, 'use_ema', False))}\n")
        f.write(f"ema_decay: {float(getattr(args, 'ema_decay', 0.999)):.6f}\n")
        f.write(f"ema_validation_start_epoch: {int(getattr(args, 'ema_validation_start_epoch', 1))}\n")
        f.write(f"lr_scheduler: {str(getattr(args, 'lr_scheduler', 'none'))}\n")
        f.write(f"lr_scheduler_monitor: {str(getattr(args, 'lr_scheduler_monitor', 'fde_m'))}\n")
        f.write(f"final_lr: {_current_lr(optimizer):.10f}\n")
        f.write("mr_metric: official_interaction_challenge_when_anchor_has_final_yaw_speed\n")
        f.write(f"fallback_miss_threshold_m: {miss_threshold_m:.6f}\n")
        f.write(f"total_train_time_s: {total_train_time_s:.6f}\n")
        f.write(f"total_val_time_s: {total_val_time_s:.6f}\n")
        f.write(f"total_epoch_time_s: {total_epoch_time_s:.6f}\n")
        f.write(f"avg_train_samples_per_s: {avg_train_samples_per_s:.6f}\n")
        f.write(f"avg_val_samples_per_s: {avg_val_samples_per_s:.6f}\n")
        f.write(f"peak_train_gpu_memory_mb: {peak_train_gpu_memory_mb:.6f}\n")
        f.write(f"peak_val_gpu_memory_mb: {peak_val_gpu_memory_mb:.6f}\n")
        f.write(f"best_checkpoint: {best_checkpoint_path}\n")
        f.write(f"best_ade_checkpoint: {best_ade_checkpoint_path}\n")
        f.write(f"best_fde_checkpoint: {best_fde_checkpoint_path}\n")
        f.write(f"best_mr_checkpoint: {best_mr_checkpoint_path}\n")
        f.write(f"last_checkpoint: {last_checkpoint_path}\n")

    if getattr(args, "save_path", None):
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_checkpoint_path, args.save_path)
        print(f"Copied best checkpoint to {args.save_path}")

    print(f"Saved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
