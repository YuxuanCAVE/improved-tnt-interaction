from __future__ import annotations

import csv
import random
import shutil
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device(device: torch.device) -> None:
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")


def create_run_dir(args: Namespace, default_root: str | Path, default_name: str) -> Path:
    output_root = Path(getattr(args, "output_root", default_root))
    run_name = getattr(args, "run_name", None) or default_name
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def copy_config(config_path: Path, run_dir: Path) -> None:
    if config_path.exists():
        shutil.copy2(config_path, run_dir / "config.yaml")


def write_metrics_csv(metrics: list[dict[str, Any]], output_path: Path) -> None:
    if not metrics:
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def write_metrics_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(metrics: list[dict[str, float]], output_path: Path) -> None:
    if not metrics:
        return
    epochs = [m["epoch"] for m in metrics]
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    axes[0].plot(epochs, [m["train_loss"] for m in metrics], label="train_loss")
    axes[0].plot(epochs, [m["val_loss"] for m in metrics], label="val_loss")
    if "aux_loss" in metrics[0]:
        axes[0].plot(epochs, [m["aux_loss"] for m in metrics], label="aux_loss")
    if "endpoint_loss" in metrics[0]:
        axes[0].plot(epochs, [m["endpoint_loss"] for m in metrics], label="endpoint_loss")
    if "final_loss" in metrics[0]:
        axes[0].plot(epochs, [m["final_loss"] for m in metrics], label="final_loss")
    if "mode_score_loss" in metrics[0]:
        axes[0].plot(epochs, [m["mode_score_loss"] for m in metrics], label="mode_score_loss")
    if "target_cls_loss" in metrics[0]:
        axes[0].plot(epochs, [m["target_cls_loss"] for m in metrics], label="target_cls_loss")
    if "target_offset_loss" in metrics[0]:
        axes[0].plot(epochs, [m["target_offset_loss"] for m in metrics], label="target_offset_loss")
    if "predicted_motion_loss" in metrics[0]:
        axes[0].plot(epochs, [m["predicted_motion_loss"] for m in metrics], label="predicted_motion_loss")
    if "endpoint_consistency_loss" in metrics[0]:
        axes[0].plot(epochs, [m["endpoint_consistency_loss"] for m in metrics], label="endpoint_consistency_loss")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [m["ade_m"] for m in metrics], color="#2563eb", label="ADE")
    axes[1].set_ylabel("ADE (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, [m["fde_m"] for m in metrics], color="#dc2626", label="FDE")
    axes[2].set_ylabel("FDE (m)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, [m["mr"] for m in metrics], color="#9333ea", label="MR")
    axes[3].set_ylabel("MR")
    axes[3].set_xlabel("epoch")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    filled = min(width, int(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_epoch_progress(
    epoch: int,
    total_epochs: int,
    batch_idx: int,
    num_batches: int,
    train_loss: float,
    elapsed: float,
    eta: float,
    val_loss: float | None = None,
    ade: float | None = None,
    fde: float | None = None,
    mr: float | None = None,
    end: bool = False,
) -> None:
    progress = 100.0 * batch_idx / max(num_batches, 1)
    val_text = "val_loss=-- ADE=-- FDE=-- MR=--"
    if val_loss is not None and ade is not None and fde is not None and mr is not None:
        val_text = f"val_loss={val_loss:.6f} ADE={ade:.3f}m FDE={fde:.3f}m MR={mr:.3f}"
    line = (
        f"epoch={epoch:03d}/{total_epochs:03d} {progress_bar(batch_idx, num_batches)} "
        f"{progress:5.1f}% train_loss={train_loss:.6f} {val_text} "
        f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
    )
    previous_len = int(getattr(print_epoch_progress, "_last_len", 0))
    padding = " " * max(0, previous_len - len(line))
    sys.stdout.write("\r" + line + padding)
    print_epoch_progress._last_len = len(line)
    if end:
        sys.stdout.write("\n")
        print_epoch_progress._last_len = 0
    sys.stdout.flush()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
