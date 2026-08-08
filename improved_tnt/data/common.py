from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class SampleRef:
    file_idx: int
    track_id: int
    end_index: int


@dataclass
class TrackFile:
    path: Path
    tracks: dict[int, pd.DataFrame]
    frames: dict[int, pd.DataFrame]
    track_frames: dict[int, dict[int, pd.Series]]


@dataclass(frozen=True)
class CoordinateNormalizer:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @classmethod
    def fit(cls, files: Iterable[TrackFile]) -> "CoordinateNormalizer":
        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")
        for track_file in files:
            for track in track_file.tracks.values():
                x = track["x"].to_numpy(np.float32)
                y = track["y"].to_numpy(np.float32)
                x_min = min(x_min, float(x.min()))
                x_max = max(x_max, float(x.max()))
                y_min = min(y_min, float(y.min()))
                y_max = max(y_max, float(y.max()))
        if not np.isfinite([x_min, x_max, y_min, y_max]).all():
            raise ValueError("Cannot fit coordinate normalizer on empty files.")
        return cls(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "CoordinateNormalizer":
        return cls(
            x_min=float(values["x_min"]),
            x_max=float(values["x_max"]),
            y_min=float(values["y_min"]),
            y_max=float(values["y_max"]),
        )

    @property
    def x_range(self) -> float:
        return max(self.x_max - self.x_min, 1e-6)

    @property
    def y_range(self) -> float:
        return max(self.y_max - self.y_min, 1e-6)

    def normalize_xy(self, xy: np.ndarray) -> np.ndarray:
        out = np.empty_like(xy, dtype=np.float32)
        out[..., 0] = (xy[..., 0] - self.x_min) / self.x_range
        out[..., 1] = (xy[..., 1] - self.y_min) / self.y_range
        return out

    def denormalize_xy_tensor(self, xy: torch.Tensor) -> torch.Tensor:
        scale = xy.new_tensor([self.x_range, self.y_range])
        offset = xy.new_tensor([self.x_min, self.y_min])
        return xy * scale + offset


def find_track_files(data_root: str | Path, scenes: Iterable[str] | None, max_files: int | None) -> list[Path]:
    root = Path(data_root)
    if scenes:
        files: list[Path] = []
        for scene in scenes:
            scene_files = sorted((root / scene).glob("vehicle_tracks_*.csv"))
            if max_files is not None:
                scene_files = scene_files[:max_files]
            files.extend(scene_files)
    else:
        files = sorted(root.glob("*/vehicle_tracks_*.csv"))
        if max_files is not None:
            files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No vehicle_tracks_*.csv files found under {root}")
    return files


def scene_name_from_track_path(path: str | Path) -> str:
    track_path = Path(path)
    parent = track_path.parent
    if parent.name in {"train", "val", "validation", "sorted", "segmented"}:
        return parent.parent.name
    return parent.name


def load_track_file(path: Path) -> TrackFile:
    usecols = [
        "track_id",
        "frame_id",
        "agent_type",
        "x",
        "y",
        "vx",
        "vy",
        "psi_rad",
        "length",
        "width",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["agent_type"].isin(["car", "truck_bus"])]
    df = df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    tracks = {int(k): g.reset_index(drop=True) for k, g in df.groupby("track_id", sort=False)}
    frames = {int(k): g.reset_index(drop=True) for k, g in df.groupby("frame_id", sort=False)}
    track_frames = {
        track_id: {int(row.frame_id): row for _, row in track.iterrows()}
        for track_id, track in tracks.items()
    }
    return TrackFile(path=path, tracks=tracks, frames=frames, track_frames=track_frames)


def rotation_to_target_frame(psi: float) -> np.ndarray:
    c = np.cos(-psi)
    s = np.sin(-psi)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def rotation_from_target_frame(psi: float) -> np.ndarray:
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


# Backward-compatible private names used by older local modules.
_load_track_file = load_track_file
_rotation_to_target_frame = rotation_to_target_frame
_rotation_from_target_frame = rotation_from_target_frame
