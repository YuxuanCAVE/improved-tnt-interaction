from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from improved_tnt.data.common import scene_name_from_track_path
from improved_tnt.visualization.maps import (
    draw_map_if_available,
    expand_view_to_include_overlay,
    set_local_view,
    set_trackfile_view,
    vehicle_polygon,
)


def draw_prediction(
    ax: plt.Axes,
    dataset,
    sample_index: int,
    pred_xy: np.ndarray,
    args: argparse.Namespace,
    title_prefix: str | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    ref = dataset.samples[sample_index]
    track_file = dataset.files[ref.file_idx]
    track = track_file.tracks[ref.track_id]

    hist_start = ref.end_index - dataset.history_steps + 1
    hist = track.iloc[hist_start : ref.end_index + 1]
    fut = track.iloc[ref.end_index + 1 : ref.end_index + dataset.future_steps + 1]
    current = track.iloc[ref.end_index]
    current_frame = track_file.frames[int(current.frame_id)]

    scene_name = scene_name_from_track_path(track_file.path)
    map_drawn = draw_map_if_available(ax, args, scene_name)
    map_xlim = ax.get_xlim()
    map_ylim = ax.get_ylim()

    ax.plot(hist["x"], hist["y"], color="#2563eb", linewidth=2.2, label="history")
    ax.plot(fut["x"], fut["y"], color="#16a34a", linewidth=2.2, label="ground truth")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#dc2626", linewidth=2.2, linestyle="--", label="prediction")
    ax.scatter([current.x], [current.y], color="#111827", s=24, zorder=40)
    ax.add_patch(
        Circle(
            (float(current.x), float(current.y)),
            dataset.sensor_range_m,
            fill=False,
            linestyle=":",
            linewidth=1.6,
            edgecolor="#7c3aed",
            label="sensor range",
            zorder=15,
        )
    )

    dx = current_frame["x"].to_numpy(np.float32) - float(current.x)
    dy = current_frame["y"].to_numpy(np.float32) - float(current.y)
    in_range = (dx * dx + dy * dy) <= dataset.sensor_range_m * dataset.sensor_range_m
    for _, row in current_frame[in_range].iterrows():
        is_target = int(row.track_id) == ref.track_id
        poly = vehicle_polygon(float(row.x), float(row.y), float(row.psi_rad), float(row.length), float(row.width))
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="#f97316" if is_target else "#94a3b8",
                edgecolor="#111827",
                alpha=0.75,
                zorder=30,
            )
        )

    if args.plot_view == "local":
        set_local_view(
            ax,
            hist[["x", "y"]].to_numpy(np.float32),
            fut[["x", "y"]].to_numpy(np.float32),
            pred_xy,
            (float(current.x), float(current.y)),
            dataset.sensor_range_m,
        )
    elif map_drawn:
        expand_view_to_include_overlay(
            ax,
            map_xlim,
            map_ylim,
            hist[["x", "y"]].to_numpy(np.float32),
            fut[["x", "y"]].to_numpy(np.float32),
            pred_xy,
            (float(current.x), float(current.y)),
            dataset.sensor_range_m,
        )
    else:
        set_trackfile_view(ax, track_file)
    ax.set_aspect("equal", adjustable="box")
    title = f"{scene_name} | track_id={ref.track_id} | sample={sample_index}"
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    if metrics is not None:
        mr_text = f"   MR={metrics['mr']:.3f}" if "mr" in metrics else ""
        ax.text(
            0.5,
            -0.15,
            f"val loss={metrics['val_loss']:.4f}   ADE={metrics['ade_m']:.3f} m   FDE={metrics['fde_m']:.3f} m{mr_text}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )


def plot_prediction(
    dataset,
    sample_index: int,
    pred_xy: np.ndarray,
    output_path: Path,
    args: argparse.Namespace,
    title_prefix: str | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_prediction(ax, dataset, sample_index, pred_xy, args, title_prefix=title_prefix, metrics=metrics)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def resolve_plot_indices(dataset, args: argparse.Namespace) -> list[int]:
    explicit_indices = getattr(args, "plot_sample_indices", None)
    if explicit_indices:
        valid_indices = []
        for index in explicit_indices:
            index = int(index)
            if index < 0 or index >= len(dataset):
                raise IndexError(f"plot_sample_indices contains invalid index {index}; dataset has {len(dataset)} samples")
            valid_indices.append(index)
        return valid_indices

    track_ids = getattr(args, "plot_track_ids", None)
    if track_ids:
        selected = []
        for track_id in track_ids:
            matches = [i for i, ref in enumerate(dataset.samples) if ref.track_id == int(track_id)]
            if not matches:
                print(f"Warning: track_id={track_id} has no valid prediction windows in this val dataset")
                continue
            selected.append(matches[len(matches) // 2])
        if selected:
            return selected

    return np.linspace(0, len(dataset) - 1, num=min(args.num_plots, len(dataset)), dtype=int).astype(int).tolist()


def plot_scenario_summary(results: list[dict], output_path: Path) -> None:
    if not results:
        return
    cols = len(results)
    fig, axes = plt.subplots(1, cols, figsize=(6.4 * cols, 6.2), squeeze=False)
    for ax, result in zip(axes[0], results):
        draw_prediction(
            ax,
            result["dataset"],
            result["sample_index"],
            result["pred_xy"],
            result.get("summary_args", result["args"]),
            title_prefix=result["name"],
            metrics=result["metrics"],
        )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
