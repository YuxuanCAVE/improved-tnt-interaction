from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from improved_tnt.data.cache import CachedPolylineDataset
from improved_tnt.data.common import scene_name_from_track_path
from improved_tnt.data.factory import polyline_validation_dataset
from improved_tnt.engine.checkpoint import load_checkpoint
from improved_tnt.engine.evaluate import evaluate_model
from improved_tnt.models import checkpoint_model_type, model_family
from improved_tnt.models.tnt import build_tnt_model
from improved_tnt.utils.config import config_namespace
from improved_tnt.utils.geometry import local_to_global_tensor
from improved_tnt.utils.io import (
    copy_config,
    create_run_dir,
    default_device,
    print_device,
    timestamp,
    to_device,
    write_metrics_table,
)
from improved_tnt.visualization.predictions import (
    plot_prediction,
    plot_scenario_summary,
    resolve_plot_indices,
)


DEFAULT_CONFIG = Path("configs/experiments/val/tnt_vectornet.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a TNT trajectory prediction checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cli_args = parser.parse_args()
    args = config_namespace(cli_args.config)
    args.config = cli_args.config
    return args


def _merged_case_args(args: argparse.Namespace, case: dict) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(case)
    return argparse.Namespace(**values)


def _validation_cases(args: argparse.Namespace) -> list[argparse.Namespace]:
    cases = getattr(args, "val_cases", None)
    if cases:
        return [_merged_case_args(args, case) for case in cases]
    scenes = list(getattr(args, "scenes", None) or [])
    if len(scenes) > 1:
        return [
            _merged_case_args(
                args,
                {
                    "name": str(scene),
                    "scenes": [str(scene)],
                    "max_files": getattr(args, "max_files", None),
                    "max_samples": getattr(args, "max_samples", None),
                    "num_plots": getattr(args, "num_plots", 0),
                    "plot_sample_indices": getattr(args, "plot_sample_indices", None),
                    "plot_track_ids": getattr(args, "plot_track_ids", None),
                    "summary_plot_view": getattr(args, "summary_plot_view", getattr(args, "plot_view", "local")),
                    "include_in_summary": True,
                },
            )
            for scene in scenes
        ]
    return [args]


def _build_tnt_checkpoint_model(checkpoint: dict, device: torch.device) -> nn.Module:
    ckpt_args = checkpoint.get("args", {})
    future_steps = int(checkpoint.get("future_steps", ckpt_args.get("future_steps")))
    model = build_tnt_model(
        input_dim=int(checkpoint["input_dim"]),
        future_steps=future_steps,
        subgraph_hidden_dim=int(ckpt_args.get("subgraph_hidden_dim", 64)),
        subgraph_layers=int(ckpt_args.get("subgraph_layers", 3)),
        graph_dim=int(ckpt_args.get("graph_dim", 128)),
        global_graph_layers=int(ckpt_args.get("global_graph_layers", 2)),
        target_hidden_dim=int(ckpt_args.get("target_hidden_dim", ckpt_args.get("endpoint_hidden_dim", 128))),
        motion_hidden_dim=int(ckpt_args.get("motion_hidden_dim", ckpt_args.get("decoder_hidden_dim", 128))),
        score_hidden_dim=int(ckpt_args.get("score_hidden_dim", ckpt_args.get("decoder_hidden_dim", 128))),
        tnt_top_m=int(ckpt_args.get("tnt_top_m", 50)),
        tnt_output_k=int(ckpt_args.get("tnt_output_k", 1)),
        target_offset_limit=float(ckpt_args.get("target_offset_limit", 0.05)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        predict_offsets=bool(ckpt_args.get("predict_offsets", False)),
        architecture=str(ckpt_args.get("architecture", checkpoint.get("architecture", "paper"))),
        use_refined_targets=bool(ckpt_args.get("use_refined_targets", True)),
        endpoint_exact_residual=bool(ckpt_args.get("endpoint_exact_residual", False)),
        trajectory_nms_threshold=float(ckpt_args.get("trajectory_nms_threshold_m", 2.0))
        / max(
            float(
                checkpoint.get(
                    "coordinate_scale_m",
                    ckpt_args.get("coordinate_scale_m") or ckpt_args.get("sensor_range_m", 1.0),
                )
            ),
            1e-6,
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model


def _build_polyline_checkpoint_model(checkpoint: dict, device: torch.device) -> nn.Module:
    """Compatibility name used by the staged TNT refinement scripts."""
    return _build_tnt_checkpoint_model(checkpoint, device)


def _loss_fn(checkpoint: dict) -> nn.Module:
    ckpt_args = checkpoint.get("args", {})
    return nn.SmoothL1Loss() if str(ckpt_args.get("loss", "mse")).lower() == "huber" else nn.MSELoss()


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


def _case_name(case_args: argparse.Namespace) -> str:
    return getattr(case_args, "name", None) or ",".join(case_args.scenes)


class _CachedSceneSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[int(index)]]

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


def _scene_names_for_cached_dataset(dataset) -> list[str]:
    if not dataset.file_paths:
        return []
    if getattr(dataset, "scene_names", None) and len(dataset.scene_names) == len(dataset.file_paths):
        return [str(name) for name in dataset.scene_names]
    return [scene_name_from_track_path(path) for path in dataset.file_paths]


def _filter_cached_dataset(dataset, case_args: argparse.Namespace) -> Dataset | _CachedSceneSubset:
    scenes = set(str(scene) for scene in getattr(case_args, "scenes", []) or [])
    if not scenes:
        if getattr(case_args, "max_samples", None) is None:
            return dataset
        limit = min(int(case_args.max_samples), len(dataset))
        return _CachedSceneSubset(dataset, list(range(limit)))
    if dataset.ref is None:
        raise ValueError("Cached validation by scene requires cache refs. Rebuild the feature cache.")
    scene_names = _scene_names_for_cached_dataset(dataset)
    indices: list[int] = []
    for sample_index, ref in enumerate(dataset.ref.tolist()):
        file_idx = int(ref[0])
        if file_idx < len(scene_names) and scene_names[file_idx] in scenes:
            indices.append(sample_index)
    if getattr(case_args, "max_samples", None) is not None:
        indices = indices[: int(case_args.max_samples)]
    if not indices:
        raise ValueError(f"No cached validation samples matched scenes={sorted(scenes)}")
    return _CachedSceneSubset(dataset, indices)


def _load_cached_polyline_dataset(args: argparse.Namespace) -> CachedPolylineDataset:
    cache_path = getattr(args, "cache_path", None)
    if cache_path is None:
        raise ValueError("use_cache=true requires cache_path in the validation config.")
    print("Using validation polyline cache:")
    print(f"  {cache_path}")
    dataset = CachedPolylineDataset(cache_path)
    expected_source = getattr(args, "target_candidate_source", None)
    if expected_source is not None:
        actual_source = str(getattr(dataset, "target_candidate_source", "all"))
        if actual_source != str(expected_source):
            raise ValueError(
                "Validation polyline cache target_candidate_source mismatch: "
                f"config expects {str(expected_source)!r}, cache contains {actual_source!r}. "
                "Rebuild the cache with precompute_cache.py."
            )
    return dataset


def _print_dataset_summary(dataset) -> None:
    if hasattr(dataset, "ref") and dataset.ref is not None:
        refs = dataset.dataset.ref[dataset.indices] if isinstance(dataset, _CachedSceneSubset) else dataset.ref
        track_ids = sorted({int(track_id) for track_id in refs[:, 1].tolist()})
    elif hasattr(dataset, "samples"):
        track_ids = sorted({ref.track_id for ref in dataset.samples})
    else:
        track_ids = []
    preview = ", ".join(str(track_id) for track_id in track_ids[:20])
    suffix = "..." if len(track_ids) > 20 else ""
    print(f"Validation samples: {len(dataset)}")
    if track_ids:
        print(f"Available target track_ids ({len(track_ids)}): {preview}{suffix}")


def _predict_polyline_sample_global(model: nn.Module, dataset, sample_index: int, device: torch.device):
    sample = dataset[int(sample_index)]
    batch = {
        "polylines": sample["polylines"].unsqueeze(0),
        "polyline_mask": sample["polyline_mask"].unsqueeze(0),
        "segment_mask": sample["segment_mask"].unsqueeze(0),
        "polyline_types": sample["polyline_types"].unsqueeze(0),
        "anchor": sample["anchor"].unsqueeze(0),
    }
    if "target_candidates" in sample and "candidate_mask" in sample:
        batch["target_candidates"] = sample["target_candidates"].unsqueeze(0)
        batch["candidate_mask"] = sample["candidate_mask"].unsqueeze(0)
    batch = to_device(batch, device)
    pred = model(
        batch["polylines"],
        batch["polyline_mask"],
        batch["segment_mask"],
        batch["polyline_types"],
        target_candidates=batch["target_candidates"],
        candidate_mask=batch["candidate_mask"],
    )
    return local_to_global_tensor(pred, batch["anchor"], dataset.coordinate_scale_m).cpu().squeeze(0).numpy()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    model_type = checkpoint_model_type(checkpoint)
    family = str(checkpoint.get("model_family", model_family(model_type)))
    if family != "polyline" or model_type != "tnt_vectornet":
        raise ValueError(f"Expected a tnt_vectornet checkpoint, got model_type={model_type!r}, family={family!r}")

    default_root = Path("runs") / model_type / "val"
    run_dir = create_run_dir(args, default_root, f"{timestamp()}_{Path(args.checkpoint).stem}")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    copy_config(args.config, run_dir)
    print(f"Run directory: {run_dir}")

    device = default_device()
    print_device(device)
    model = _build_tnt_checkpoint_model(checkpoint, device)
    if getattr(args, "tnt_output_k", None) is not None:
        model.tnt_output_k = int(args.tnt_output_k)
    if getattr(args, "tnt_top_m", None) is not None:
        model.tnt_top_m = int(args.tnt_top_m)
    if getattr(args, "trajectory_nms_threshold_m", None) is not None:
        ckpt_args = checkpoint.get("args", {})
        scale_m = float(checkpoint.get("coordinate_scale_m", ckpt_args.get("coordinate_scale_m") or ckpt_args.get("sensor_range_m", 1.0)))
        model.trajectory_nms_threshold = float(args.trajectory_nms_threshold_m) / max(scale_m, 1e-6)

    loss_fn = _loss_fn(checkpoint)
    miss_threshold_m = float(
        getattr(args, "miss_threshold_m", None)
        or checkpoint.get("miss_threshold_m", checkpoint.get("args", {}).get("miss_threshold_m", 2.0))
    )
    if getattr(args, "save_dir", None):
        args.save_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    summary_results: list[dict] = []
    cached_dataset = _load_cached_polyline_dataset(args) if bool(getattr(args, "use_cache", False)) else None

    for case_no, case_args in enumerate(_validation_cases(args), start=1):
        case_name = _case_name(case_args)
        print(f"Validation case {case_no}: {case_name}")
        if cached_dataset is not None:
            dataset = _filter_cached_dataset(cached_dataset, case_args)
            if int(case_args.num_plots) > 0:
                raise ValueError("Cached validation does not support plots. Set num_plots: 0 or disable use_cache.")
        else:
            dataset = polyline_validation_dataset(case_args, checkpoint)
        _print_dataset_summary(dataset)

        loader = DataLoader(
            dataset,
            batch_size=int(case_args.batch_size),
            shuffle=False,
            num_workers=int(case_args.num_workers),
            pin_memory=torch.cuda.is_available(),
        )
        _sync_if_cuda(device)
        _reset_peak_memory_if_cuda(device)
        val_start = time.perf_counter()
        val_loss, ade, fde, mr = evaluate_model(
            family,
            model,
            loader,
            loss_fn,
            device,
            dataset,
            miss_threshold_m=miss_threshold_m,
        )
        _sync_if_cuda(device)
        val_time_s = time.perf_counter() - val_start
        val_samples_per_s = len(dataset) / max(val_time_s, 1e-9)
        val_peak_gpu_memory_mb = _peak_memory_mb(device)
        print(f"{case_name}: val_loss={val_loss:.6f} ADE={ade:.3f}m FDE={fde:.3f}m MR={mr:.3f}")
        print(
            "compute: "
            f"val_time={val_time_s:.2f}s val_samples_per_s={val_samples_per_s:.2f} "
            f"val_peak_gpu_memory={_format_memory_mb(val_peak_gpu_memory_mb)}"
        )
        metrics = {
            "val_loss": val_loss,
            "ade_m": ade,
            "fde_m": fde,
            "mr": mr,
            "num_samples": len(dataset),
            "val_time_s": val_time_s,
            "val_samples_per_s": val_samples_per_s,
            "val_peak_gpu_memory_mb": val_peak_gpu_memory_mb,
        }
        rows.append(
            {
                "case": case_name,
                "scenes": ",".join(case_args.scenes),
                "val_loss": val_loss,
                "ade_m": ade,
                "fde_m": fde,
                "mr": mr,
                "miss_threshold_m": miss_threshold_m,
                "num_samples": len(dataset),
                "val_time_s": val_time_s,
                "val_samples_per_s": val_samples_per_s,
                "val_peak_gpu_memory_mb": val_peak_gpu_memory_mb,
            }
        )

        if int(case_args.num_plots) <= 0:
            continue
        plot_indices = resolve_plot_indices(dataset, case_args)
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_name)
        with torch.no_grad():
            for plot_no, sample_index in enumerate(plot_indices, start=1):
                pred_m = _predict_polyline_sample_global(model, dataset, int(sample_index), device)
                if plot_no == 1 and bool(getattr(case_args, "include_in_summary", True)):
                    summary_args = argparse.Namespace(**vars(case_args))
                    summary_args.plot_view = getattr(case_args, "summary_plot_view", case_args.plot_view)
                    summary_results.append(
                        {
                            "name": case_name,
                            "dataset": dataset,
                            "sample_index": int(sample_index),
                            "pred_xy": pred_m,
                            "args": case_args,
                            "summary_args": summary_args,
                            "metrics": metrics,
                        }
                    )
                filename = f"{safe_name}_prediction_{plot_no:03d}_sample_{int(sample_index):06d}.png"
                output_path = plots_dir / filename
                plot_prediction(dataset, int(sample_index), pred_m, output_path, case_args, case_name, metrics)
                print(f"saved {output_path}")
                if getattr(case_args, "save_dir", None):
                    case_args.save_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_path, case_args.save_dir / filename)

    write_metrics_table(rows, run_dir / "metrics.csv")
    total_samples = sum(int(row["num_samples"]) for row in rows)
    if total_samples > 0:
        overall_ade = sum(float(row["ade_m"]) * int(row["num_samples"]) for row in rows) / total_samples
        overall_fde = sum(float(row["fde_m"]) * int(row["num_samples"]) for row in rows) / total_samples
        overall_mr = sum(float(row["mr"]) * int(row["num_samples"]) for row in rows) / total_samples
        overall_line = f"ADE = {overall_ade:.6f}, FDE = {overall_fde:.6f}, MR = {overall_mr:.6f}"
        print(overall_line)
    else:
        overall_line = "ADE = nan, FDE = nan, MR = nan"
    total_val_time_s = sum(float(row["val_time_s"]) for row in rows)
    overall_val_samples_per_s = total_samples / max(total_val_time_s, 1e-9) if total_samples else 0.0
    peak_val_gpu_memory_mb = max((float(row["val_peak_gpu_memory_mb"]) for row in rows), default=0.0)
    compute_line = (
        f"total_val_time_s = {total_val_time_s:.6f}, "
        f"overall_val_samples_per_s = {overall_val_samples_per_s:.6f}, "
        f"peak_val_gpu_memory_mb = {peak_val_gpu_memory_mb:.6f}"
    )
    print(compute_line)

    if summary_results:
        summary_path = run_dir / "scenario_summary.png"
        plot_scenario_summary(summary_results, summary_path)
        print(f"saved {summary_path}")
        if getattr(args, "save_dir", None):
            shutil.copy2(summary_path, args.save_dir / "scenario_summary.png")

    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        for row in rows:
            f.write(
                f"{row['case']}: val_loss={float(row['val_loss']):.6f}, "
                f"ade_m={float(row['ade_m']):.6f}, fde_m={float(row['fde_m']):.6f}, "
                f"mr={float(row['mr']):.6f}, miss_threshold_m={float(row['miss_threshold_m']):.6f}, "
                f"num_samples={int(row['num_samples'])}, val_time_s={float(row['val_time_s']):.6f}, "
                f"val_samples_per_s={float(row['val_samples_per_s']):.6f}, "
                f"val_peak_gpu_memory_mb={float(row['val_peak_gpu_memory_mb']):.6f}\n"
            )
        f.write(f"{overall_line}\n")
        f.write(f"{compute_line}\n")

    print(f"Saved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
