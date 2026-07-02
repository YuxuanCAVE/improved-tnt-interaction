from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path
import time

import torch

from improved_tnt.data.common import scene_name_from_track_path
from improved_tnt.data.factory import find_track_files_by_scene
from improved_tnt.data.polyline import InteractionPolylineDataset
from improved_tnt.utils.config import config_namespace


DEFAULT_CONFIG = Path("configs/experiments/train/polyline/tnt_vectornet.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute TNT polyline tensor caches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=500)
    cli_args = parser.parse_args()
    cfg = config_namespace(cli_args.config)
    cfg.config = cli_args.config
    cfg.output = cli_args.output
    cfg.log_interval = cli_args.log_interval
    return cfg


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _truthy_path(value: object) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "none", "null"}


def _copy_args_with(args: argparse.Namespace, **overrides: object) -> argparse.Namespace:
    task_args = copy(args)
    for key, value in overrides.items():
        setattr(task_args, key, value)
    return task_args


def _cache_tasks(args: argparse.Namespace) -> list[tuple[str, argparse.Namespace, Path]]:
    if args.output is not None:
        return [("cache", args, Path(args.output))]

    tasks: list[tuple[str, argparse.Namespace, Path]] = []
    if bool(getattr(args, "use_cache", False)) and _truthy_path(getattr(args, "cache_path", None)):
        tasks.append(("train", args, Path(args.cache_path)))

    val_use_cache = bool(getattr(args, "val_use_cache", False))
    if val_use_cache and _truthy_path(getattr(args, "val_cache_path", None)):
        val_args = _copy_args_with(
            args,
            data_split=getattr(args, "val_data_split", None) or "val",
            scenes=getattr(args, "val_scenes", getattr(args, "scenes", None)),
            max_files=getattr(args, "val_max_files", getattr(args, "max_files", None)),
            max_samples=getattr(args, "val_max_samples", None),
            samples_per_scene_type=getattr(
                args, "val_samples_per_scene_type", getattr(args, "samples_per_scene_type", None)
            ),
            samples_per_scenario=getattr(
                args, "val_samples_per_scenario", getattr(args, "samples_per_scenario", None)
            ),
        )
        tasks.append(("val", val_args, Path(args.val_cache_path)))

    if not tasks and _truthy_path(getattr(args, "cache_path", None)):
        tasks.append(("cache", args, Path(args.cache_path)))

    if not tasks:
        raise ValueError("Set cache_path in the YAML, pass --output, or enable use_cache/val_use_cache.")
    return tasks


def _build_dataset(args: argparse.Namespace) -> tuple[InteractionPolylineDataset, list[Path]]:
    files = find_track_files_by_scene(
        args.data_root,
        args.scenes,
        args.max_files,
        data_split=getattr(args, "data_split", None),
    )
    dataset = InteractionPolylineDataset(
        files,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        stride=args.stride,
        num_neighbors=args.num_neighbors,
        sensor_range_m=args.sensor_range_m,
        coordinate_scale_m=getattr(args, "coordinate_scale_m", None),
        map_root=getattr(args, "map_root", None),
        map_range_m=getattr(args, "map_range_m", None),
        max_map_polylines=args.max_map_polylines,
        max_segments=args.max_segments,
        include_map=bool(args.include_map),
        include_velocity=bool(getattr(args, "include_velocity", True)),
        max_target_candidates=int(getattr(args, "max_target_candidates", 0) or 0),
        target_candidate_spacing_m=float(getattr(args, "target_candidate_spacing_m", 2.0)),
        target_candidate_nearest_fraction=float(getattr(args, "target_candidate_nearest_fraction", 0.5)),
        target_candidate_source=str(getattr(args, "target_candidate_source", "all") or "all"),
        target_candidate_lateral_offsets_m=getattr(args, "target_candidate_lateral_offsets_m", None),
        target_candidate_grid_spacing_m=float(getattr(args, "target_candidate_grid_spacing_m", 2.0)),
        target_candidate_lane_quota=getattr(args, "target_candidate_lane_quota", None),
        target_candidate_grid_quota=getattr(args, "target_candidate_grid_quota", None),
        target_candidate_selection=str(
            getattr(args, "target_candidate_selection", "nearest_plus_radial_angular_coverage")
        ),
        target_candidate_reachable_dt_s=float(getattr(args, "target_candidate_reachable_dt_s", 0.1)),
        target_candidate_reachable_primary_fraction=float(
            getattr(args, "target_candidate_reachable_primary_fraction", 0.6)
        ),
        max_samples=args.max_samples,
        samples_per_scene_type=getattr(args, "samples_per_scene_type", None),
        samples_per_scenario=getattr(args, "samples_per_scenario", None),
        sampling_seed=int(getattr(args, "seed", 7)),
    )
    return dataset, files


def _precompute_polyline(args: argparse.Namespace, output: Path) -> None:
    dataset, files = _build_dataset(args)
    print(f"Building polyline cache from {len(files)} CSV file(s)")

    tensors: dict[str, list[torch.Tensor]] = {
        "polylines": [],
        "polyline_mask": [],
        "segment_mask": [],
        "polyline_types": [],
        "target": [],
        "anchor": [],
        "ref": [],
    }
    if getattr(dataset, "max_target_candidates", 0) > 0:
        tensors["target_candidates"] = []
        tensors["candidate_mask"] = []

    start = time.perf_counter()
    for index in range(len(dataset)):
        item = dataset[index]
        for key in tensors:
            tensors[key].append(item[key])
        done = index + 1
        if done == 1 or done == len(dataset) or done % max(1, args.log_interval) == 0:
            elapsed = time.perf_counter() - start
            rate = done / max(elapsed, 1e-9)
            eta = (len(dataset) - done) / max(rate, 1e-9)
            print(
                f"  cached {done}/{len(dataset)} ({100.0 * done / len(dataset):5.1f}%) "
                f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
            )

    cache = {key: torch.stack(values) for key, values in tensors.items()}
    cache["meta"] = {
        "history_steps": dataset.history_steps,
        "future_steps": dataset.future_steps,
        "stride": dataset.stride,
        "num_neighbors": dataset.num_neighbors,
        "sensor_range_m": dataset.sensor_range_m,
        "coordinate_scale_m": dataset.coordinate_scale_m,
        "map_range_m": dataset.map_range_m,
        "max_map_polylines": dataset.max_map_polylines,
        "max_segments": dataset.max_segments,
        "max_polylines": dataset.max_polylines,
        "feature_dim": dataset.feature_dim,
        "target_mode": dataset.target_mode,
        "include_map": dataset.include_map,
        "include_velocity": dataset.include_velocity,
        "max_target_candidates": dataset.max_target_candidates,
        "target_candidate_spacing_m": dataset.target_candidate_spacing_m,
        "target_candidate_nearest_fraction": dataset.target_candidate_nearest_fraction,
        "target_candidate_source": dataset.target_candidate_source,
        "target_candidate_type_ids": (
            list(dataset.target_candidate_type_ids) if dataset.target_candidate_type_ids is not None else None
        ),
        "target_candidate_lateral_offsets_m": list(dataset.target_candidate_lateral_offsets_m),
        "target_candidate_grid_spacing_m": dataset.target_candidate_grid_spacing_m,
        "target_candidate_lane_quota": dataset.target_candidate_lane_quota,
        "target_candidate_grid_quota": dataset.target_candidate_grid_quota,
        "target_candidate_selection": dataset.target_candidate_selection,
        "target_candidate_reachable_dt_s": dataset.target_candidate_reachable_dt_s,
        "target_candidate_reachable_primary_fraction": dataset.target_candidate_reachable_primary_fraction,
        "max_samples": args.max_samples,
        "samples_per_scene_type": getattr(args, "samples_per_scene_type", None),
        "samples_per_scenario": getattr(args, "samples_per_scenario", None),
        "data_split": getattr(args, "data_split", None),
        "sampling_seed": int(getattr(args, "seed", 7)),
        "scenes": list(args.scenes) if getattr(args, "scenes", None) is not None else None,
        "file_paths": [str(path) for path in files],
        "scene_names": [scene_name_from_track_path(path) for path in files],
        "num_files": len(files),
    }
    torch.save(cache, output)


def main() -> None:
    args = parse_args()
    for task_idx, (label, task_args, output) in enumerate(_cache_tasks(args), start=1):
        output.parent.mkdir(parents=True, exist_ok=True)
        split = getattr(task_args, "data_split", None)
        print(f"[{task_idx}] Building {label} cache: split={split}, output={output}")
        _precompute_polyline(task_args, output)
        print(f"Saved cache to {output}")


if __name__ == "__main__":
    main()
