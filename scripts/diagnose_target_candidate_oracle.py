from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from improved_tnt.data.common import _rotation_to_target_frame, scene_name_from_track_path
from improved_tnt.data.factory import find_track_files_by_scene
from improved_tnt.data.polyline import (
    INTERACTION_LANE_BOUNDARY_TYPE_IDS,
    InteractionPolylineDataset,
)


@dataclass(frozen=True)
class CandidateVariant:
    name: str
    max_candidates: int
    spacing_m: float
    nearest_fraction: float
    lateral_offsets_m: tuple[float, ...] = (0.0,)
    include_lane_boundaries: bool = True
    include_grid: bool = False
    grid_range_m: float = 100.0
    grid_spacing_m: float = 2.0
    lane_quota: int | None = None
    grid_quota: int | None = None
    selection_strategy: str = "default"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _sample_polyline_points(
    points: np.ndarray,
    spacing_m: float,
    lateral_offsets_m: Iterable[float],
    map_range_m: float,
) -> list[np.ndarray]:
    sampled: list[np.ndarray] = []
    spacing = max(float(spacing_m), 1e-3)
    offsets = tuple(float(value) for value in lateral_offsets_m)
    for start, end in zip(points[:-1], points[1:]):
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-3:
            continue
        direction = vector / length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        count = max(1, int(math.ceil(length / spacing)))
        for step in range(count + 1):
            base = start + vector * (step / count)
            for offset in offsets:
                point = base + normal * offset
                if float(np.linalg.norm(point)) <= map_range_m:
                    sampled.append(point.astype(np.float32))
    return sampled


def _grid_points(range_m: float, spacing_m: float, map_range_m: float) -> list[np.ndarray]:
    limit = min(float(range_m), float(map_range_m))
    spacing = max(float(spacing_m), 1e-3)
    coords = np.arange(-limit, limit + 0.5 * spacing, spacing, dtype=np.float32)
    points: list[np.ndarray] = []
    for x in coords:
        for y in coords:
            point = np.array([x, y], dtype=np.float32)
            if float(np.linalg.norm(point)) <= map_range_m:
                points.append(point)
    return points


def _stack_points(points: list[np.ndarray]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(points).astype(np.float32)


def _dedupe_points(points: Iterable[np.ndarray], quant_m: float) -> list[np.ndarray]:
    quant = max(float(quant_m), 1e-3)
    unique: dict[tuple[int, int], np.ndarray] = {}
    for point in points:
        key = (int(round(float(point[0]) / quant)), int(round(float(point[1]) / quant)))
        unique.setdefault(key, point.astype(np.float32))
    return list(unique.values())


def _select_points(
    points: list[np.ndarray],
    max_candidates: int,
    nearest_fraction: float,
    map_range_m: float,
) -> list[np.ndarray]:
    if max_candidates <= 0:
        return points
    if len(points) <= max_candidates:
        return sorted(points, key=lambda point: float(np.linalg.norm(point)))

    max_count = int(max_candidates)
    nearest_count = int(round(max_count * float(np.clip(nearest_fraction, 0.0, 1.0))))
    nearest_count = min(max(nearest_count, 0), max_count)
    ordered = sorted(points, key=lambda point: float(np.linalg.norm(point)))
    selected = ordered[:nearest_count]
    selected_ids = {id(point) for point in selected}

    angle_bins = 32
    radial_bins = 8
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for point in ordered:
        if id(point) in selected_ids:
            continue
        distance = float(np.linalg.norm(point))
        angle = float(np.arctan2(point[1], point[0]))
        radial_bin = min(radial_bins - 1, int(distance / max(float(map_range_m), 1e-3) * radial_bins))
        angle_bin = min(angle_bins - 1, int((angle + np.pi) / (2.0 * np.pi) * angle_bins))
        buckets.setdefault((radial_bin, angle_bin), []).append(point)

    bucket_keys = sorted(buckets.keys(), key=lambda key: (key[0], key[1]))
    while len(selected) < max_count and bucket_keys:
        next_keys: list[tuple[int, int]] = []
        for key in bucket_keys:
            bucket = buckets[key]
            if bucket and len(selected) < max_count:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        bucket_keys = next_keys
    return selected


def _oracle_distance_m(points: list[np.ndarray], target_endpoint_m: np.ndarray) -> float:
    if not points:
        return float("nan")
    stacked = np.stack(points).astype(np.float32)
    return float(np.linalg.norm(stacked - target_endpoint_m[None, :], axis=1).min())


def _oracle_distance_array_m(points: np.ndarray, target_endpoint_m: np.ndarray) -> float:
    if points.size == 0:
        return float("nan")
    return float(np.linalg.norm(points - target_endpoint_m[None, :], axis=1).min())


def _dedupe_array(points: np.ndarray, quant_m: float) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 2).astype(np.float32)
    quant = max(float(quant_m), 1e-3)
    keys = np.rint(points / quant).astype(np.int32)
    _, first_idx = np.unique(keys, axis=0, return_index=True)
    first_idx.sort()
    return points[first_idx].astype(np.float32)


def _select_array(
    points: np.ndarray,
    max_candidates: int,
    nearest_fraction: float,
    map_range_m: float,
) -> np.ndarray:
    if max_candidates <= 0 or points.shape[0] <= max_candidates:
        return points

    max_count = int(max_candidates)
    distances = np.linalg.norm(points, axis=1)
    nearest_count = int(round(max_count * float(np.clip(nearest_fraction, 0.0, 1.0))))
    nearest_count = min(max(nearest_count, 0), max_count)
    selected_indices: list[int] = []
    selected_mask = np.zeros(points.shape[0], dtype=bool)
    if nearest_count > 0:
        nearest_idx = np.argpartition(distances, nearest_count - 1)[:nearest_count]
        nearest_idx = nearest_idx[np.argsort(distances[nearest_idx])]
        selected_indices.extend(int(idx) for idx in nearest_idx)
        selected_mask[nearest_idx] = True

    remaining_idx = np.nonzero(~selected_mask)[0]
    if remaining_idx.size == 0:
        return points[np.asarray(selected_indices, dtype=np.int64)]

    remaining = points[remaining_idx]
    remaining_dist = distances[remaining_idx]
    angles = np.arctan2(remaining[:, 1], remaining[:, 0])
    radial_bins = 8
    angle_bins = 32
    radial = np.minimum(radial_bins - 1, (remaining_dist / max(float(map_range_m), 1e-3) * radial_bins).astype(np.int32))
    angle = np.minimum(angle_bins - 1, ((angles + np.pi) / (2.0 * np.pi) * angle_bins).astype(np.int32))

    buckets: dict[tuple[int, int], list[int]] = {}
    order = np.argsort(remaining_dist)
    for pos in order:
        key = (int(radial[pos]), int(angle[pos]))
        buckets.setdefault(key, []).append(int(remaining_idx[pos]))

    bucket_keys = sorted(buckets.keys(), key=lambda key: (key[0], key[1]))
    while len(selected_indices) < max_count and bucket_keys:
        next_keys: list[tuple[int, int]] = []
        for key in bucket_keys:
            bucket = buckets[key]
            if bucket and len(selected_indices) < max_count:
                selected_indices.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        bucket_keys = next_keys
    return points[np.asarray(selected_indices, dtype=np.int64)]


def _select_reachable_array(
    points: np.ndarray,
    max_candidates: int,
    map_range_m: float,
    current_velocity_mps: np.ndarray,
    horizon_s: float,
) -> np.ndarray:
    if max_candidates <= 0 or points.shape[0] <= max_candidates:
        return points

    max_count = int(max_candidates)
    speed = float(np.linalg.norm(current_velocity_mps))
    cv_endpoint = current_velocity_mps.astype(np.float32) * float(horizon_s)
    dist_to_cv = np.linalg.norm(points - cv_endpoint[None, :], axis=1)
    dist_to_origin = np.linalg.norm(points, axis=1)

    if speed > 1e-3:
        direction = current_velocity_mps / speed
        longitudinal = points @ direction
        lateral_vec = points - longitudinal[:, None] * direction[None, :]
        lateral = np.linalg.norm(lateral_vec, axis=1)
        backward_penalty = np.maximum(-longitudinal, 0.0) * 2.0
    else:
        lateral = np.zeros(points.shape[0], dtype=np.float32)
        backward_penalty = np.zeros(points.shape[0], dtype=np.float32)

    reach_radius = max(15.0, speed * float(horizon_s) + 0.5 * 3.0 * float(horizon_s) ** 2 + 8.0)
    over_reach_penalty = np.maximum(dist_to_origin - reach_radius, 0.0) * 1.5
    score = dist_to_cv + 0.15 * lateral + 0.05 * dist_to_origin + backward_penalty + over_reach_penalty

    primary_count = min(max_count, int(round(max_count * 0.6)))
    primary_idx = np.argpartition(score, primary_count - 1)[:primary_count]
    primary_idx = primary_idx[np.argsort(score[primary_idx])]

    selected_indices = [int(idx) for idx in primary_idx]
    selected_mask = np.zeros(points.shape[0], dtype=bool)
    selected_mask[primary_idx] = True
    remaining = points[~selected_mask]
    remaining_idx = np.nonzero(~selected_mask)[0]
    if len(selected_indices) >= max_count or remaining.size == 0:
        return points[np.asarray(selected_indices[:max_count], dtype=np.int64)]

    # Fill the rest with radial/angle coverage so turns and low-speed cases are not collapsed to CV-only points.
    remaining_selected = _select_array(
        remaining,
        max_count - len(selected_indices),
        nearest_fraction=0.0,
        map_range_m=map_range_m,
    )
    if remaining_selected.size:
        selected_lookup = {tuple(point.tolist()) for point in remaining_selected}
        for idx, point in zip(remaining_idx, remaining):
            if tuple(point.tolist()) in selected_lookup:
                selected_indices.append(int(idx))
                if len(selected_indices) >= max_count:
                    break
    return points[np.asarray(selected_indices[:max_count], dtype=np.int64)]


def _build_variant_array(
    dataset: InteractionPolylineDataset,
    scene_name: str,
    anchor_pos: np.ndarray,
    rot: np.ndarray,
    variant: CandidateVariant,
    world_candidate_cache: dict[tuple[str, str], np.ndarray],
    grid_array_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    arrays: list[np.ndarray] = []
    if variant.include_lane_boundaries:
        world_points = world_candidate_cache[(scene_name, variant.name)]
        if world_points.size:
            local_points = (world_points - anchor_pos) @ rot.T
            keep = np.linalg.norm(local_points, axis=1) <= dataset.map_range_m
            arrays.append(local_points[keep].astype(np.float32))

    if variant.include_grid:
        arrays.append(grid_array_cache[variant.name])

    if not arrays:
        return np.zeros((0, 2), dtype=np.float32), 0
    points = np.concatenate(arrays, axis=0)
    return points, int(points.shape[0])


def _build_selected_variant_array(
    dataset: InteractionPolylineDataset,
    scene_name: str,
    anchor_pos: np.ndarray,
    rot: np.ndarray,
    variant: CandidateVariant,
    world_candidate_cache: dict[tuple[str, str], np.ndarray],
    grid_array_cache: dict[str, np.ndarray],
    current_velocity_mps: np.ndarray | None = None,
    horizon_s: float = 3.0,
) -> tuple[np.ndarray, int]:
    points, _raw_count = _build_variant_array(
        dataset,
        scene_name,
        anchor_pos,
        rot,
        variant,
        world_candidate_cache,
        grid_array_cache,
    )
    deduped = _dedupe_array(points, min(variant.spacing_m, max(variant.grid_spacing_m, 1e-3)))
    if variant.selection_strategy == "reachable" and current_velocity_mps is not None:
        selected = _select_reachable_array(
            deduped,
            variant.max_candidates,
            dataset.map_range_m,
            current_velocity_mps,
            horizon_s,
        )
    else:
        selected = _select_array(
            deduped,
            variant.max_candidates,
            variant.nearest_fraction,
            dataset.map_range_m,
        )
    return selected, int(deduped.shape[0])


def _build_quota_variant_array(
    dataset: InteractionPolylineDataset,
    scene_name: str,
    anchor_pos: np.ndarray,
    rot: np.ndarray,
    variant: CandidateVariant,
    world_candidate_cache: dict[tuple[str, str], np.ndarray],
    grid_array_cache: dict[str, np.ndarray],
    current_velocity_mps: np.ndarray | None = None,
    horizon_s: float = 3.0,
) -> tuple[np.ndarray, int]:
    selected_parts: list[np.ndarray] = []
    total_count = 0

    lane_quota = int(variant.lane_quota or 0)
    if lane_quota > 0:
        lane_world = world_candidate_cache[(scene_name, variant.name)]
        if lane_world.size:
            lane_local = (lane_world - anchor_pos) @ rot.T
            keep = np.linalg.norm(lane_local, axis=1) <= dataset.map_range_m
            lane_points = _dedupe_array(
                lane_local[keep].astype(np.float32),
                min(variant.spacing_m, max(variant.grid_spacing_m, 1e-3)),
            )
        else:
            lane_points = np.zeros((0, 2), dtype=np.float32)
        total_count += int(lane_points.shape[0])
        if variant.selection_strategy == "reachable" and current_velocity_mps is not None:
            selected_parts.append(
                _select_reachable_array(lane_points, lane_quota, dataset.map_range_m, current_velocity_mps, horizon_s)
            )
        else:
            selected_parts.append(_select_array(lane_points, lane_quota, variant.nearest_fraction, dataset.map_range_m))

    grid_quota = int(variant.grid_quota or 0)
    if grid_quota > 0:
        grid_points = _dedupe_array(
            grid_array_cache[variant.name],
            min(variant.spacing_m, max(variant.grid_spacing_m, 1e-3)),
        )
        total_count += int(grid_points.shape[0])
        if variant.selection_strategy == "reachable" and current_velocity_mps is not None:
            selected_parts.append(
                _select_reachable_array(grid_points, grid_quota, dataset.map_range_m, current_velocity_mps, horizon_s)
            )
        else:
            selected_parts.append(_select_array(grid_points, grid_quota, variant.nearest_fraction, dataset.map_range_m))

    if not selected_parts:
        return np.zeros((0, 2), dtype=np.float32), total_count
    selected = np.concatenate([part for part in selected_parts if part.size], axis=0)
    selected = _dedupe_array(selected, min(variant.spacing_m, max(variant.grid_spacing_m, 1e-3)))
    if selected.shape[0] > variant.max_candidates:
        selected = _select_array(selected, variant.max_candidates, 0.0, dataset.map_range_m)
    return selected, total_count




def _build_variant_points(
    dataset: InteractionPolylineDataset,
    scene_name: str,
    anchor_pos: np.ndarray,
    rot: np.ndarray,
    variant: CandidateVariant,
    world_candidate_cache: dict[tuple[str, str], np.ndarray],
    grid_candidate_cache: dict[str, list[np.ndarray]],
) -> tuple[list[np.ndarray], int]:
    points: list[np.ndarray] = []
    if variant.include_lane_boundaries:
        world_points = world_candidate_cache[(scene_name, variant.name)]
        if world_points.size:
            local_points = (world_points - anchor_pos) @ rot.T
            keep = np.linalg.norm(local_points, axis=1) <= dataset.map_range_m
            points.extend([point.astype(np.float32) for point in local_points[keep]])

    if variant.include_grid:
        points.extend(grid_candidate_cache[variant.name])

    deduped = _dedupe_points(points, min(variant.spacing_m, max(variant.grid_spacing_m, 1e-3)))
    selected = _select_points(
        deduped,
        variant.max_candidates,
        variant.nearest_fraction,
        dataset.map_range_m,
    )
    return selected, len(deduped)


def _precompute_world_candidates(
    dataset: InteractionPolylineDataset,
    scenes: Iterable[str],
    variants: Iterable[CandidateVariant],
) -> dict[tuple[str, str], np.ndarray]:
    cache: dict[tuple[str, str], np.ndarray] = {}
    for scene_name in scenes:
        for variant in variants:
            if not variant.include_lane_boundaries:
                cache[(scene_name, variant.name)] = np.zeros((0, 2), dtype=np.float32)
                continue
            points: list[np.ndarray] = []
            for map_polyline in dataset._scene_map_polylines(scene_name):
                if map_polyline.type_id not in INTERACTION_LANE_BOUNDARY_TYPE_IDS:
                    continue
                points.extend(
                    _sample_polyline_points(
                        map_polyline.points,
                        variant.spacing_m,
                        variant.lateral_offsets_m,
                        map_range_m=float("inf"),
                    )
                )
            deduped = _dedupe_points(points, min(variant.spacing_m, max(variant.grid_spacing_m, 1e-3)))
            cache[(scene_name, variant.name)] = _stack_points(deduped)
    return cache


def _precompute_grid_candidates(variants: Iterable[CandidateVariant], map_range_m: float) -> dict[str, list[np.ndarray]]:
    cache: dict[str, list[np.ndarray]] = {}
    for variant in variants:
        if variant.include_grid:
            cache[variant.name] = _grid_points(variant.grid_range_m, variant.grid_spacing_m, map_range_m)
        else:
            cache[variant.name] = []
    return cache


def _precompute_grid_arrays(variants: Iterable[CandidateVariant], map_range_m: float) -> dict[str, np.ndarray]:
    return {name: _stack_points(points) for name, points in _precompute_grid_candidates(variants, map_range_m).items()}


def _summarize(values: list[float], candidate_counts: list[int]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "samples": len(values),
            "valid_samples": 0,
            "mean_m": float("nan"),
            "median_m": float("nan"),
            "p90_m": float("nan"),
            "p95_m": float("nan"),
            "max_m": float("nan"),
            "over1m": float("nan"),
            "over2m": float("nan"),
            "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        }
    return {
        "samples": len(values),
        "valid_samples": int(finite.size),
        "mean_m": float(np.mean(finite)),
        "median_m": float(np.median(finite)),
        "p90_m": float(np.percentile(finite, 90)),
        "p95_m": float(np.percentile(finite, 95)),
        "max_m": float(np.max(finite)),
        "over1m": float(np.mean(finite > 1.0)),
        "over2m": float(np.mean(finite > 2.0)),
        "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }


def _scenario_type(scene_name: str) -> str:
    lowered = scene_name.lower()
    if "roundabout" in lowered:
        return "roundabout"
    if "intersection" in lowered:
        return "intersection"
    if "merging" in lowered:
        return "merging"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute target-candidate oracle coverage by scene.")
    parser.add_argument(
        "--config",
        default="configs/experiments/train/polyline/tnt_vectornet.yaml",
        help="Training or validation YAML used for data paths and sample indexing.",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Which split to diagnose.")
    parser.add_argument("--max-samples-per-scene", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None, help="Override selected candidate count for every variant.")
    parser.add_argument(
        "--profile",
        choices=["default", "second_stage"],
        default="default",
        help="Use 'second_stage' for A-E candidate sampler diagnostics.",
    )
    parser.add_argument("--dt-s", type=float, default=0.1, help="Frame step in seconds for reachable endpoint scoring.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["all", "selected"],
        default="all",
        help="'all' measures candidate coverage before top-N truncation; 'selected' applies the cache-style selector.",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    split_prefix = "val_" if args.split == "val" else ""
    data_split = cfg.get(f"{split_prefix}data_split", cfg.get("data_split", "train"))
    data_root = Path(cfg["data_root"])
    scenes = cfg.get(f"{split_prefix}scenes", cfg.get("scenes"))
    max_files = cfg.get(f"{split_prefix}max_files", cfg.get("max_files"))
    max_samples = cfg.get(f"{split_prefix}max_samples", cfg.get("max_samples"))
    samples_per_scene_type = cfg.get(
        f"{split_prefix}samples_per_scene_type",
        cfg.get("samples_per_scene_type"),
    )
    samples_per_scenario = cfg.get(
        f"{split_prefix}samples_per_scenario",
        cfg.get("samples_per_scenario"),
    )
    if args.max_samples_per_scene is not None:
        samples_per_scenario = args.max_samples_per_scene
        max_samples = None

    track_files = find_track_files_by_scene(data_root, scenes, max_files, data_split=data_split)
    dataset = InteractionPolylineDataset(
        track_files,
        history_steps=int(cfg.get("history_steps", 10)),
        future_steps=int(cfg.get("future_steps", 30)),
        stride=int(cfg.get("stride", 10)),
        num_neighbors=int(cfg.get("num_neighbors", 8)),
        sensor_range_m=float(cfg.get("sensor_range_m", 100.0)),
        coordinate_scale_m=cfg.get("coordinate_scale_m"),
        map_root=cfg.get("map_root"),
        map_range_m=float(cfg.get("map_range_m", cfg.get("sensor_range_m", 100.0))),
        max_map_polylines=int(cfg.get("max_map_polylines", 96)),
        max_segments=int(cfg.get("max_segments", 30)),
        include_map=True,
        include_velocity=True,
        max_target_candidates=0,
        max_samples=max_samples,
        samples_per_scene_type=samples_per_scene_type,
        samples_per_scenario=samples_per_scenario,
        sampling_seed=int(cfg.get("seed", 7)),
    )

    base_max = int(args.max_candidates or cfg.get("max_target_candidates", 1024) or 1024)
    base_spacing = float(cfg.get("target_candidate_spacing_m", 1.0) or 1.0)
    base_nearest = float(cfg.get("target_candidate_nearest_fraction", 0.5) or 0.5)
    map_range_m = float(cfg.get("map_range_m", cfg.get("sensor_range_m", 100.0)))
    if args.profile == "second_stage":
        selected_max = int(args.max_candidates or 2048)
        lane_grid_quota = selected_max // 2
        variants = [
            CandidateVariant(
                name="A_current_lane_boundaries_1024",
                max_candidates=int(cfg.get("max_target_candidates", 1024) or 1024),
                spacing_m=base_spacing,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(0.0,),
                include_lane_boundaries=True,
                include_grid=False,
            ),
            CandidateVariant(
                name="B_lateral_offsets_2048",
                max_candidates=selected_max,
                spacing_m=1.0,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(-2.0, -1.0, 0.0, 1.0, 2.0),
                include_lane_boundaries=True,
                include_grid=False,
            ),
            CandidateVariant(
                name="C_local_grid_2048",
                max_candidates=selected_max,
                spacing_m=1.0,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(),
                include_lane_boundaries=False,
                include_grid=True,
                grid_range_m=map_range_m,
                grid_spacing_m=2.0,
            ),
            CandidateVariant(
                name="D_quota_hybrid_1024_lane_1024_grid",
                max_candidates=selected_max,
                spacing_m=1.0,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(-2.0, -1.0, 0.0, 1.0, 2.0),
                include_lane_boundaries=True,
                include_grid=True,
                grid_range_m=map_range_m,
                grid_spacing_m=2.0,
                lane_quota=lane_grid_quota,
                grid_quota=selected_max - lane_grid_quota,
            ),
            CandidateVariant(
                name="E_reachable_quota_hybrid_1024_lane_1024_grid",
                max_candidates=selected_max,
                spacing_m=1.0,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(-2.0, -1.0, 0.0, 1.0, 2.0),
                include_lane_boundaries=True,
                include_grid=True,
                grid_range_m=map_range_m,
                grid_spacing_m=2.0,
                lane_quota=lane_grid_quota,
                grid_quota=selected_max - lane_grid_quota,
                selection_strategy="reachable",
            ),
        ]
    else:
        variants = [
            CandidateVariant(
                name="raw_lane_boundaries",
                max_candidates=base_max,
                spacing_m=base_spacing,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(0.0,),
                include_lane_boundaries=True,
                include_grid=False,
            ),
            CandidateVariant(
                name="lane_boundaries_lateral_offsets",
                max_candidates=max(base_max, 2048),
                spacing_m=1.0,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(-2.0, -1.0, 0.0, 1.0, 2.0),
                include_lane_boundaries=True,
                include_grid=False,
            ),
            CandidateVariant(
                name="local_grid",
                max_candidates=max(base_max, 2048),
                spacing_m=1.0,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(),
                include_lane_boundaries=False,
                include_grid=True,
                grid_range_m=map_range_m,
                grid_spacing_m=2.0,
            ),
            CandidateVariant(
                name="hybrid",
                max_candidates=max(base_max, 2048),
                spacing_m=1.0,
                nearest_fraction=base_nearest,
                lateral_offsets_m=(-2.0, -1.0, 0.0, 1.0, 2.0),
                include_lane_boundaries=True,
                include_grid=True,
                grid_range_m=map_range_m,
                grid_spacing_m=2.0,
            ),
        ]
    scene_names = sorted({scene_name_from_track_path(path) for path in track_files})
    print(f"Precomputing candidates for {len(scene_names)} scene(s)")
    world_candidate_cache = _precompute_world_candidates(dataset, scene_names, variants)
    grid_candidate_cache = _precompute_grid_candidates(variants, dataset.map_range_m)
    grid_array_cache = {name: _stack_points(points) for name, points in grid_candidate_cache.items()}

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs/diagnostics/target_candidate_oracle") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample.csv"
    summary_path = output_dir / "summary.csv"
    scene_summary_path = output_dir / "summary_by_scene_type.csv"

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    candidate_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    type_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    type_candidate_counts: dict[tuple[str, str], list[int]] = defaultdict(list)

    with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "scene_type",
                "file_idx",
                "track_id",
                "end_index",
                "target_x_m",
                "target_y_m",
                "variant",
                "oracle_distance_m",
                "candidate_count",
            ],
        )
        writer.writeheader()

        for sample_idx, ref in enumerate(dataset.samples):
            track_file = dataset.files[ref.file_idx]
            track = track_file.tracks[ref.track_id]
            anchor = track.iloc[ref.end_index]
            future = track.iloc[ref.end_index + 1 : ref.end_index + dataset.future_steps + 1]
            if len(future) < dataset.future_steps:
                continue

            anchor_pos = np.array([anchor.x, anchor.y], dtype=np.float32)
            rot = _rotation_to_target_frame(float(anchor.psi_rad))
            scene = scene_name_from_track_path(track_file.path)
            scene_type = _scenario_type(scene)
            endpoint = future[["x", "y"]].to_numpy(np.float32)[-1]
            target_endpoint_m = (endpoint - anchor_pos) @ rot.T
            current_velocity_mps = rot @ np.array([float(anchor.vx), float(anchor.vy)], dtype=np.float32)
            horizon_s = float(dataset.future_steps) * float(args.dt_s)

            for variant in variants:
                if args.selection_mode == "selected":
                    if variant.lane_quota is not None or variant.grid_quota is not None:
                        point_array, count = _build_quota_variant_array(
                            dataset,
                            scene,
                            anchor_pos,
                            rot,
                            variant,
                            world_candidate_cache,
                            grid_array_cache,
                            current_velocity_mps=current_velocity_mps,
                            horizon_s=horizon_s,
                        )
                    else:
                        point_array, count = _build_selected_variant_array(
                            dataset,
                            scene,
                            anchor_pos,
                            rot,
                            variant,
                            world_candidate_cache,
                            grid_array_cache,
                            current_velocity_mps=current_velocity_mps,
                            horizon_s=horizon_s,
                        )
                    dist = _oracle_distance_array_m(point_array, target_endpoint_m)
                else:
                    point_array, count = _build_variant_array(
                        dataset,
                        scene,
                        anchor_pos,
                        rot,
                        variant,
                        world_candidate_cache,
                        grid_array_cache,
                    )
                    dist = _oracle_distance_array_m(point_array, target_endpoint_m)
                key = (scene, variant.name)
                values[key].append(dist)
                candidate_counts[key].append(count)
                type_key = (scene_type, variant.name)
                type_values[type_key].append(dist)
                type_candidate_counts[type_key].append(count)
                writer.writerow(
                    {
                        "scene": scene,
                        "scene_type": scene_type,
                        "file_idx": ref.file_idx,
                        "track_id": ref.track_id,
                        "end_index": ref.end_index,
                        "target_x_m": float(target_endpoint_m[0]),
                        "target_y_m": float(target_endpoint_m[1]),
                        "variant": variant.name,
                        "oracle_distance_m": dist,
                        "candidate_count": count,
                    }
                )

            if (sample_idx + 1) % 1000 == 0:
                print(f"Processed {sample_idx + 1}/{len(dataset.samples)} samples")

    fieldnames = [
        "group",
        "variant",
        "samples",
        "valid_samples",
        "mean_m",
        "median_m",
        "p90_m",
        "p95_m",
        "max_m",
        "over1m",
        "over2m",
        "candidate_count_mean",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (scene, variant), distances in sorted(values.items()):
            row = {"group": scene, "variant": variant}
            row.update(_summarize(distances, candidate_counts[(scene, variant)]))
            writer.writerow(row)

    with scene_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (scene_type, variant), distances in sorted(type_values.items()):
            row = {"group": scene_type, "variant": variant}
            row.update(_summarize(distances, type_candidate_counts[(scene_type, variant)]))
            writer.writerow(row)

    print(f"Wrote {summary_path}")
    print(f"Wrote {scene_summary_path}")
    print(f"Wrote {per_sample_path}")


if __name__ == "__main__":
    main()
