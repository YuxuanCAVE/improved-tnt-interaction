from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from improved_tnt.data.cache import CachedPolylineDataset
from improved_tnt.data.common import (
    CoordinateNormalizer,
    find_track_files,
    scene_name_from_track_path,
)
from improved_tnt.data.polyline import InteractionPolylineDataset, TARGET_CANDIDATE_META_KEYS
from improved_tnt.models import model_family


def find_track_files_by_scene(
    data_root: str | Path,
    scenes: list[str] | None,
    max_files: int | None,
    data_split: str | None = None,
) -> list[Path]:
    root = Path(data_root)
    if not scenes:
        if data_split:
            files = sorted(root.glob(f"*/{data_split}/vehicle_tracks_*.csv"))
            if max_files is not None:
                files = files[:max_files]
            if not files:
                raise FileNotFoundError(
                    f"No vehicle_tracks_*.csv files found under {root} with data_split={data_split!r}"
                )
            return files
        return find_track_files(data_root, scenes, max_files)
    files: list[Path] = []
    for scene in scenes:
        search_dir = root / scene / data_split if data_split else root / scene
        scene_files = sorted(search_dir.glob("vehicle_tracks_*.csv"))
        if max_files is not None:
            scene_files = scene_files[:max_files]
        files.extend(scene_files)
    if not files:
        raise FileNotFoundError(f"No vehicle_tracks_*.csv files found under {root} with data_split={data_split!r}")
    return files


def configured_cache_paths(args: Namespace) -> list[Path]:
    cache_paths = getattr(args, "cache_paths", None)
    if cache_paths is None:
        cache_paths = [getattr(args, "cache_path")]
    return [Path(path) for path in cache_paths]


def validate_cache_paths(paths: list[Path], model_family_name: str, config_path: Path) -> None:
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        cache_script = "precompute_cache.py"
        raise FileNotFoundError(
            "Cache file(s) not found: "
            + ", ".join(str(path) for path in missing_paths)
            + f". Run: python {cache_script} --config {config_path}"
        )


def build_training_dataset(args: Namespace, normalizer: CoordinateNormalizer | None = None):
    family = model_family(str(getattr(args, "model_type", "lstm")))
    if family != "polyline":
        raise ValueError("This repository contains only the TNT polyline training pipeline.")
    use_cache = bool(getattr(args, "use_cache", False))
    if use_cache:
        cache_paths = configured_cache_paths(args)
        validate_cache_paths(cache_paths, family, args.config)
        print("Using tensor cache:")
        for path in cache_paths:
            print(f"  {path}")
        dataset = CachedPolylineDataset(cache_paths)
        for key in TARGET_CANDIDATE_META_KEYS:
            if not hasattr(args, key):
                continue
            expected = getattr(args, key)
            if expected is None:
                continue
            actual = getattr(dataset, key, None)
            if isinstance(expected, list):
                expected = tuple(float(value) for value in expected)
            if isinstance(actual, list):
                actual = tuple(float(value) for value in actual)
            if isinstance(actual, tuple) and all(isinstance(value, (float, int)) for value in actual):
                if isinstance(expected, tuple):
                    expected = tuple(float(value) for value in expected)
                else:
                    expected = (float(expected),)
            if actual != expected:
                raise ValueError(
                    f"Polyline cache {key} mismatch: config expects {expected!r}, "
                    f"cache contains {actual!r}. Rebuild the cache with precompute_cache.py."
                )
        return dataset

    files = find_track_files_by_scene(
        args.data_root,
        args.scenes,
        args.max_files,
        data_split=getattr(args, "data_split", None),
    )
    print(f"Using {len(files)} CSV file(s)")
    for path in files:
        print(f"  {path}")

    return InteractionPolylineDataset(
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


def print_training_dataset_summary(dataset: Any, args: Namespace) -> None:
    if hasattr(dataset, "normalizer"):
        print(
            "Coordinate normalizer: "
            f"x=[{dataset.normalizer.x_min:.3f}, {dataset.normalizer.x_max:.3f}], "
            f"y=[{dataset.normalizer.y_min:.3f}, {dataset.normalizer.y_max:.3f}]"
        )
    if hasattr(dataset, "max_polylines"):
        print(
            f"Dataset windows: total={len(dataset)}; "
            f"polylines={dataset.max_polylines}, max_segments={dataset.max_segments}, "
            f"feature_dim={dataset.feature_dim}, include_map={dataset.include_map}"
        )
        print(
            f"Dynamic agents limited to {dataset.sensor_range_m:.1f} m; "
            f"map polylines limited to {dataset.map_range_m:.1f} m"
        )
    else:
        print(f"Surrounding vehicles are limited to {dataset.sensor_range_m:.1f} m")

    has_sample_refs = hasattr(dataset, "samples") and hasattr(dataset, "files")
    if (
        not bool(getattr(args, "use_cache", False))
        and has_sample_refs
        and getattr(args, "samples_per_scene_type", None) is not None
    ):
        counts: dict[str, int] = {}
        for ref in dataset.samples:
            scene_type = dataset.scene_type_from_name(scene_name_from_track_path(dataset.files[ref.file_idx].path))
            counts[scene_type] = counts.get(scene_type, 0) + 1
        print("Samples per scene type: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))

    if (
        not bool(getattr(args, "use_cache", False))
        and has_sample_refs
        and getattr(args, "samples_per_scenario", None) is not None
    ):
        counts: dict[str, int] = {}
        for ref in dataset.samples:
            scene_name = scene_name_from_track_path(dataset.files[ref.file_idx].path)
            counts[scene_name] = counts.get(scene_name, 0) + 1
        print("Samples per scenario: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


def polyline_validation_dataset(case_args: Namespace, checkpoint: dict[str, Any]) -> InteractionPolylineDataset:
    ckpt_args = checkpoint.get("args", {})
    history_steps = int(_arg_or_checkpoint(case_args, ckpt_args, "history_steps"))
    future_steps = int(checkpoint.get("future_steps", _arg_or_checkpoint(case_args, ckpt_args, "future_steps")))
    stride = int(_arg_or_checkpoint(case_args, ckpt_args, "stride"))
    num_neighbors = int(_arg_or_checkpoint(case_args, ckpt_args, "num_neighbors"))
    sensor_range_m = float(_arg_or_checkpoint(case_args, ckpt_args, "sensor_range_m"))
    coordinate_scale_m = float(
        checkpoint.get("coordinate_scale_m", ckpt_args.get("coordinate_scale_m") or sensor_range_m)
    )
    files = find_track_files_by_scene(
        case_args.data_root,
        case_args.scenes,
        case_args.max_files,
        data_split=getattr(case_args, "data_split", None),
    )
    return InteractionPolylineDataset(
        files,
        history_steps=history_steps,
        future_steps=future_steps,
        stride=stride,
        num_neighbors=num_neighbors,
        sensor_range_m=sensor_range_m,
        coordinate_scale_m=coordinate_scale_m,
        map_root=getattr(case_args, "map_root", None),
        map_range_m=getattr(case_args, "map_range_m", ckpt_args.get("map_range_m", None)),
        max_map_polylines=int(ckpt_args.get("max_map_polylines", getattr(case_args, "max_map_polylines", 64))),
        max_segments=int(ckpt_args.get("max_segments", getattr(case_args, "max_segments", 20))),
        include_map=bool(ckpt_args.get("include_map", getattr(case_args, "include_map", True))),
        include_velocity=bool(ckpt_args.get("include_velocity", checkpoint.get("include_velocity", True))),
        max_target_candidates=int(ckpt_args.get("max_target_candidates", getattr(case_args, "max_target_candidates", 0) or 0)),
        target_candidate_spacing_m=float(
            ckpt_args.get("target_candidate_spacing_m", getattr(case_args, "target_candidate_spacing_m", 2.0))
        ),
        target_candidate_nearest_fraction=float(
            ckpt_args.get(
                "target_candidate_nearest_fraction",
                getattr(case_args, "target_candidate_nearest_fraction", 0.5),
            )
        ),
        target_candidate_source=str(
            ckpt_args.get(
                "target_candidate_source",
                getattr(case_args, "target_candidate_source", "all"),
            )
            or "all"
        ),
        target_candidate_lateral_offsets_m=ckpt_args.get(
            "target_candidate_lateral_offsets_m",
            getattr(case_args, "target_candidate_lateral_offsets_m", None),
        ),
        target_candidate_grid_spacing_m=float(
            ckpt_args.get(
                "target_candidate_grid_spacing_m",
                getattr(case_args, "target_candidate_grid_spacing_m", 2.0),
            )
        ),
        target_candidate_lane_quota=ckpt_args.get(
            "target_candidate_lane_quota",
            getattr(case_args, "target_candidate_lane_quota", None),
        ),
        target_candidate_grid_quota=ckpt_args.get(
            "target_candidate_grid_quota",
            getattr(case_args, "target_candidate_grid_quota", None),
        ),
        target_candidate_selection=str(
            ckpt_args.get(
                "target_candidate_selection",
                getattr(case_args, "target_candidate_selection", "nearest_plus_radial_angular_coverage"),
            )
        ),
        target_candidate_reachable_dt_s=float(
            ckpt_args.get(
                "target_candidate_reachable_dt_s",
                getattr(case_args, "target_candidate_reachable_dt_s", 0.1),
            )
        ),
        target_candidate_reachable_primary_fraction=float(
            ckpt_args.get(
                "target_candidate_reachable_primary_fraction",
                getattr(case_args, "target_candidate_reachable_primary_fraction", 0.6),
            )
        ),
        max_samples=case_args.max_samples,
        samples_per_scene_type=None,
    )


def arg_or_checkpoint(args: Namespace, ckpt_args: dict[str, Any], name: str):
    return _arg_or_checkpoint(args, ckpt_args, name)


def _arg_or_checkpoint(args: Namespace, ckpt_args: dict[str, Any], name: str):
    value = getattr(args, name, None)
    return value if value is not None else ckpt_args[name]
