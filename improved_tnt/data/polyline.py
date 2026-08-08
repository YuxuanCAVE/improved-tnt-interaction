from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from improved_tnt.data.common import (
    SampleRef,
    TrackFile,
    _load_track_file,
    _rotation_to_target_frame,
    scene_name_from_track_path,
)


VEL_SCALE = 30.0
TARGET_CANDIDATE_META_KEYS = (
    "max_target_candidates",
    "target_candidate_spacing_m",
    "target_candidate_nearest_fraction",
    "target_candidate_source",
    "target_candidate_type_ids",
    "target_candidate_lateral_offsets_m",
    "target_candidate_grid_spacing_m",
    "target_candidate_lane_quota",
    "target_candidate_grid_quota",
    "target_candidate_selection",
    "target_candidate_reachable_dt_s",
    "target_candidate_reachable_primary_fraction",
)


MAP_TYPE_TO_ID = {
    "line_thin": 2,
    "line_thick": 3,
    "curbstone": 4,
    "road_border": 5,
    "guard_rail": 6,
    "stop_line": 7,
    "virtual": 8,
    "pedestrian_marking": 9,
    "bike_marking": 10,
}

INTERACTION_LANE_BOUNDARY_TYPE_IDS = frozenset(
    MAP_TYPE_TO_ID[name]
    for name in ("line_thin", "line_thick", "curbstone", "road_border", "guard_rail", "virtual")
)


@dataclass(frozen=True)
class MapPolyline:
    points: np.ndarray
    type_id: int


def target_candidate_type_ids(source: str | None) -> tuple[int, ...] | None:
    source_name = str(source or "all").lower()
    if source_name in {"all", "map", "all_map"}:
        return None
    if source_name in {
        "lane_boundary",
        "lane_boundaries",
        "interaction_lane_boundaries",
        "lane_boundaries_lateral_offsets",
        "lateral_offsets",
        "quota_hybrid",
        "hybrid_interaction",
        "reachable_quota_hybrid",
    }:
        return tuple(sorted(INTERACTION_LANE_BOUNDARY_TYPE_IDS))
    if source_name in {"local_grid", "grid"}:
        return None
    raise ValueError(
        f"Unsupported target_candidate_source={source!r}. "
        "Use 'all', 'lane_boundaries', 'local_grid', or 'reachable_quota_hybrid'."
    )


def _tag_value(element: ET.Element, key: str) -> str | None:
    for tag in element.findall("tag"):
        if tag.get("k") == key:
            return tag.get("v")
    return None


def _load_osm_xy_polylines(path: Path) -> list[MapPolyline]:
    root = ET.parse(path).getroot()
    nodes: dict[int, np.ndarray] = {}
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        nodes[node_id] = np.array([float(node.get("x")), float(node.get("y"))], dtype=np.float32)

    polylines: list[MapPolyline] = []
    for way in root.findall("way"):
        way_type = _tag_value(way, "type")
        if way_type not in MAP_TYPE_TO_ID:
            continue
        points = [nodes[int(nd.get("ref"))] for nd in way.findall("nd") if int(nd.get("ref")) in nodes]
        if len(points) < 2:
            continue
        polylines.append(MapPolyline(points=np.stack(points).astype(np.float32), type_id=MAP_TYPE_TO_ID[way_type]))
    return polylines


class InteractionPolylineDataset(Dataset):
    """INTERACTION windows represented as VectorNet-style polylines.

    The dataset returns fixed-size padded tensors so the default PyTorch
    DataLoader can batch samples without a custom collate function.

    Returned keys:
        polylines: [P, S, F] vector segment features.
        polyline_mask: [P] true for valid polylines.
        segment_mask: [P, S] true for valid vector segments.
        target: [future_steps, 2] future ego-local trajectory / coordinate_scale_m.
        anchor: [5] global anchor x/y/yaw plus evaluated target GT final yaw/speed for official MR.
        ref: [3] file_idx, track_id, end_index.
    """

    def __init__(
        self,
        csv_files: Iterable[str | Path],
        history_steps: int = 10,
        future_steps: int = 30,
        stride: int = 5,
        num_neighbors: int = 8,
        sensor_range_m: float = 50.0,
        coordinate_scale_m: float | None = None,
        map_root: str | Path | None = None,
        map_range_m: float | None = None,
        max_map_polylines: int = 64,
        max_segments: int | None = None,
        include_map: bool = True,
        include_velocity: bool = True,
        max_target_candidates: int = 0,
        target_candidate_spacing_m: float = 2.0,
        target_candidate_nearest_fraction: float = 0.5,
        target_candidate_source: str | None = "all",
        target_candidate_lateral_offsets_m: Sequence[float] | None = None,
        target_candidate_grid_spacing_m: float = 2.0,
        target_candidate_lane_quota: int | None = None,
        target_candidate_grid_quota: int | None = None,
        target_candidate_selection: str = "nearest_plus_radial_angular_coverage",
        target_candidate_reachable_dt_s: float = 0.1,
        target_candidate_reachable_primary_fraction: float = 0.6,
        max_samples: int | None = None,
        samples_per_scene_type: int | None = None,
        samples_per_scenario: int | None = None,
        sampling_seed: int = 7,
    ) -> None:
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.stride = stride
        self.num_neighbors = num_neighbors
        self.sensor_range_m = sensor_range_m
        self.coordinate_scale_m = float(coordinate_scale_m or sensor_range_m)
        self.map_root = Path(map_root) if map_root is not None else None
        self.map_range_m = float(map_range_m or sensor_range_m)
        self.max_map_polylines = max_map_polylines
        self.max_segments = int(max_segments or max(history_steps - 1, 20))
        self.include_map = include_map and self.map_root is not None
        self.include_velocity = include_velocity
        self.max_target_candidates = int(max_target_candidates or 0)
        self.target_candidate_spacing_m = float(target_candidate_spacing_m)
        self.target_candidate_nearest_fraction = float(np.clip(target_candidate_nearest_fraction, 0.0, 1.0))
        self.target_candidate_source = str(target_candidate_source or "all")
        self.target_candidate_type_ids = target_candidate_type_ids(self.target_candidate_source)
        self.target_candidate_lateral_offsets_m = tuple(
            float(value)
            for value in (
                target_candidate_lateral_offsets_m
                if target_candidate_lateral_offsets_m is not None
                else (0.0,)
            )
        )
        self.target_candidate_grid_spacing_m = float(target_candidate_grid_spacing_m)
        self.target_candidate_lane_quota = (
            int(target_candidate_lane_quota) if target_candidate_lane_quota is not None else None
        )
        self.target_candidate_grid_quota = (
            int(target_candidate_grid_quota) if target_candidate_grid_quota is not None else None
        )
        self.target_candidate_selection = str(target_candidate_selection or "nearest_plus_radial_angular_coverage")
        self.target_candidate_reachable_dt_s = float(target_candidate_reachable_dt_s)
        self.target_candidate_reachable_primary_fraction = float(
            np.clip(target_candidate_reachable_primary_fraction, 0.0, 1.0)
        )

        self.agent_polyline_slots = 1 + num_neighbors
        self.max_polylines = self.agent_polyline_slots + max_map_polylines
        self.feature_dim = 12 if include_velocity else 8
        self.target_mode = "ego_local"
        self.samples_per_scene_type = samples_per_scene_type
        self.samples_per_scenario = samples_per_scenario
        self.sampling_seed = sampling_seed

        self.files = [_load_track_file(Path(p)) for p in csv_files]
        self.samples = self._build_sample_index(max_samples, samples_per_scene_type, samples_per_scenario)
        self._map_cache: dict[str, list[MapPolyline]] = {}
        self._target_map_point_cache: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}
        self._target_grid_cache: dict[tuple[float, float], list[np.ndarray]] = {}

    @staticmethod
    def scene_type_from_name(scene_name: str) -> str:
        lowered = scene_name.lower()
        if "merging" in lowered:
            return "merging"
        if "intersection" in lowered:
            return "intersection"
        if "roundabout" in lowered:
            return "roundabout"
        return scene_name

    def _build_sample_index(
        self,
        max_samples: int | None,
        samples_per_scene_type: int | None,
        samples_per_scenario: int | None,
    ) -> list[SampleRef]:
        samples: list[SampleRef] = []
        samples_by_type: dict[str, list[SampleRef]] = {}
        samples_by_scenario: dict[str, list[SampleRef]] = {}
        total_steps = self.history_steps + self.future_steps
        for file_idx, track_file in enumerate(self.files):
            scene_name = scene_name_from_track_path(track_file.path)
            scene_type = self.scene_type_from_name(scene_name)
            for track_id, track in track_file.tracks.items():
                if len(track) < total_steps:
                    continue
                frame_ids = track["frame_id"].to_numpy()
                contiguous = np.r_[True, np.diff(frame_ids) == 1]
                for end_index in range(self.history_steps - 1, len(track) - self.future_steps, self.stride):
                    start = end_index - self.history_steps + 1
                    stop = end_index + self.future_steps + 1
                    if contiguous[start:stop].all():
                        ref = SampleRef(file_idx, track_id, end_index)
                        if samples_per_scenario is None and samples_per_scene_type is None:
                            samples.append(ref)
                            if max_samples is not None and len(samples) >= max_samples:
                                return samples
                        elif samples_per_scenario is not None:
                            samples_by_scenario.setdefault(scene_name, []).append(ref)
                        else:
                            samples_by_type.setdefault(scene_type, []).append(ref)
        if samples_per_scenario is not None:
            rng = random.Random(self.sampling_seed)
            for scene_name in sorted(samples_by_scenario):
                scene_samples = samples_by_scenario[scene_name]
                rng.shuffle(scene_samples)
                samples.extend(scene_samples[:samples_per_scenario])
            rng.shuffle(samples)
            if max_samples is not None:
                samples = samples[:max_samples]
        elif samples_per_scene_type is not None:
            rng = random.Random(self.sampling_seed)
            for scene_type in sorted(samples_by_type):
                scene_samples = samples_by_type[scene_type]
                rng.shuffle(scene_samples)
                samples.extend(scene_samples[:samples_per_scene_type])
            rng.shuffle(samples)
            if max_samples is not None:
                samples = samples[:max_samples]
        if not samples:
            raise ValueError("No valid trajectory windows found. Try smaller history/future steps.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _scene_map_polylines(self, scene_name: str) -> list[MapPolyline]:
        if not self.include_map or self.map_root is None:
            return []
        if scene_name not in self._map_cache:
            map_path = self.map_root / f"{scene_name}.osm_xy"
            if not map_path.exists():
                self._map_cache[scene_name] = []
            else:
                self._map_cache[scene_name] = _load_osm_xy_polylines(map_path)
        return self._map_cache[scene_name]

    def _selected_neighbor_ids(self, track_file: TrackFile, target_track_id: int, anchor: pd.Series) -> list[int]:
        frame = track_file.frames[int(anchor.frame_id)]
        neighbors = frame[frame["track_id"] != target_track_id]
        if len(neighbors) == 0:
            return []

        dx = neighbors["x"].to_numpy(np.float32) - float(anchor.x)
        dy = neighbors["y"].to_numpy(np.float32) - float(anchor.y)
        dist_sq = dx * dx + dy * dy
        in_range = dist_sq <= self.sensor_range_m * self.sensor_range_m
        if not in_range.any():
            return []
        order = np.argsort(dist_sq[in_range])[: self.num_neighbors]
        selected = neighbors.iloc[np.flatnonzero(in_range)[order]]
        return [int(track_id) for track_id in selected["track_id"].tolist()]

    def _empty_output(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        polylines = np.zeros((self.max_polylines, self.max_segments, self.feature_dim), dtype=np.float32)
        polyline_mask = np.zeros((self.max_polylines,), dtype=bool)
        segment_mask = np.zeros((self.max_polylines, self.max_segments), dtype=bool)
        polyline_types = np.zeros((self.max_polylines,), dtype=np.int64)
        return polylines, polyline_mask, segment_mask, polyline_types

    def _segment_feature(
        self,
        start: np.ndarray,
        end: np.ndarray,
        type_id: int,
        start_vel: np.ndarray | None = None,
        end_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        delta = end - start
        features = [
            start[0] / self.coordinate_scale_m,
            start[1] / self.coordinate_scale_m,
            end[0] / self.coordinate_scale_m,
            end[1] / self.coordinate_scale_m,
            delta[0] / self.coordinate_scale_m,
            delta[1] / self.coordinate_scale_m,
        ]
        if self.include_velocity:
            if start_vel is None:
                start_vel = np.zeros(2, dtype=np.float32)
            if end_vel is None:
                end_vel = np.zeros(2, dtype=np.float32)
            features.extend(
                [
                    start_vel[0] / VEL_SCALE,
                    start_vel[1] / VEL_SCALE,
                    end_vel[0] / VEL_SCALE,
                    end_vel[1] / VEL_SCALE,
                ]
            )
        features.extend([float(type_id) / 10.0, 1.0])
        return np.array(features, dtype=np.float32)

    def _write_agent_polyline(
        self,
        polylines: np.ndarray,
        polyline_mask: np.ndarray,
        segment_mask: np.ndarray,
        polyline_types: np.ndarray,
        slot: int,
        rows: list[pd.Series],
        anchor_pos: np.ndarray,
        rot: np.ndarray,
        type_id: int,
    ) -> None:
        if len(rows) < 2 or slot >= self.max_polylines:
            return
        max_count = min(len(rows) - 1, self.max_segments)
        for seg_idx in range(max_count):
            p0 = np.array([rows[seg_idx].x, rows[seg_idx].y], dtype=np.float32)
            p1 = np.array([rows[seg_idx + 1].x, rows[seg_idx + 1].y], dtype=np.float32)
            start = rot @ (p0 - anchor_pos)
            end = rot @ (p1 - anchor_pos)
            v0 = rot @ np.array([rows[seg_idx].vx, rows[seg_idx].vy], dtype=np.float32)
            v1 = rot @ np.array([rows[seg_idx + 1].vx, rows[seg_idx + 1].vy], dtype=np.float32)
            polylines[slot, seg_idx] = self._segment_feature(start, end, type_id, v0, v1)
            segment_mask[slot, seg_idx] = True
        polyline_mask[slot] = True
        polyline_types[slot] = type_id

    def _write_map_polylines(
        self,
        polylines: np.ndarray,
        polyline_mask: np.ndarray,
        segment_mask: np.ndarray,
        polyline_types: np.ndarray,
        scene_name: str,
        anchor_pos: np.ndarray,
        rot: np.ndarray,
    ) -> None:
        map_polylines = self._scene_map_polylines(scene_name)
        if not map_polylines:
            return

        candidates: list[tuple[float, MapPolyline, np.ndarray, int, int]] = []
        for map_polyline in map_polylines:
            local_points = (map_polyline.points - anchor_pos) @ rot.T
            segment_count = local_points.shape[0] - 1
            if segment_count <= 0:
                continue

            closest_segment, min_dist = self._closest_segment_to_origin(local_points)
            if min_dist <= self.map_range_m:
                window_count = min(segment_count, self.max_segments)
                window_start = max(0, closest_segment - window_count // 2)
                window_start = min(window_start, segment_count - window_count)
                candidates.append(
                    (
                        min_dist,
                        map_polyline,
                        local_points.astype(np.float32),
                        int(window_start),
                        int(window_count),
                    )
                )
        candidates.sort(key=lambda item: item[0])

        for map_idx, (_dist, map_polyline, local_points, window_start, window_count) in enumerate(
            candidates[: self.max_map_polylines]
        ):
            slot = self.agent_polyline_slots + map_idx
            if window_count <= 0:
                continue
            for seg_idx in range(window_count):
                point_idx = window_start + seg_idx
                polylines[slot, seg_idx] = self._segment_feature(
                    local_points[point_idx],
                    local_points[point_idx + 1],
                    map_polyline.type_id,
                )
                segment_mask[slot, seg_idx] = True
            polyline_mask[slot] = True
            polyline_types[slot] = map_polyline.type_id

    @staticmethod
    def _closest_segment_to_origin(points: np.ndarray) -> tuple[int, float]:
        starts = points[:-1]
        ends = points[1:]
        vectors = ends - starts
        lengths_sq = np.sum(vectors * vectors, axis=1)

        projection = np.zeros_like(lengths_sq, dtype=np.float32)
        valid = lengths_sq > 1e-6
        projection[valid] = -np.sum(starts[valid] * vectors[valid], axis=1) / lengths_sq[valid]
        projection = np.clip(projection, 0.0, 1.0)

        closest_points = starts + vectors * projection[:, None]
        distances = np.linalg.norm(closest_points, axis=1)
        closest_segment = int(np.argmin(distances))
        return closest_segment, float(distances[closest_segment])

    def _target_candidates(
        self,
        scene_name: str,
        anchor_pos: np.ndarray,
        rot: np.ndarray,
        anchor_vel: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.zeros((self.max_target_candidates, 2), dtype=np.float32)
        candidate_mask = np.zeros((self.max_target_candidates,), dtype=bool)
        if self.max_target_candidates <= 0:
            return candidates, candidate_mask

        selected = self._target_candidate_points(scene_name, anchor_pos, rot, anchor_vel)
        if selected:
            count = min(len(selected), self.max_target_candidates)
            candidates[:count] = np.stack(selected[:count]).astype(np.float32) / self.coordinate_scale_m
            candidate_mask[:count] = True
        return candidates, candidate_mask

    def _target_candidate_points(
        self,
        scene_name: str,
        anchor_pos: np.ndarray,
        rot: np.ndarray,
        anchor_vel: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        source = self.target_candidate_source.lower()
        if source == "reachable_quota_hybrid":
            return self._reachable_quota_hybrid_target_candidates(scene_name, anchor_pos, rot, anchor_vel)
        if source in {"quota_hybrid", "hybrid_interaction"}:
            return self._quota_hybrid_target_candidates(scene_name, anchor_pos, rot, anchor_vel=None)

        points: list[np.ndarray] = []
        if source in {"local_grid", "grid"}:
            points.extend(self._local_grid_target_candidates())
        else:
            offsets = self._target_candidate_offsets_for_source(source)
            points.extend(self._map_target_candidate_points(scene_name, anchor_pos, rot, offsets))
            if source in {"hybrid", "hybrid_interaction"}:
                points.extend(self._local_grid_target_candidates())

        if not points:
            fallback_x = np.linspace(0.0, self.map_range_m, self.max_target_candidates, dtype=np.float32)
            points = [np.array([x, 0.0], dtype=np.float32) for x in fallback_x]

        unique = self._dedupe_target_candidates(points)
        return self._select_target_candidates(unique)[: self.max_target_candidates]

    def _target_candidate_offsets_for_source(self, source: str) -> tuple[float, ...]:
        if source in {
            "lane_boundaries_lateral_offsets",
            "lateral_offsets",
            "quota_hybrid",
            "hybrid",
            "hybrid_interaction",
            "reachable_quota_hybrid",
        }:
            return self.target_candidate_lateral_offsets_m
        return (0.0,)

    def _map_target_candidate_points(
        self,
        scene_name: str,
        anchor_pos: np.ndarray,
        rot: np.ndarray,
        lateral_offsets_m: Sequence[float],
    ) -> list[np.ndarray]:
        offsets = tuple(float(value) for value in lateral_offsets_m)
        world_points = self._world_map_target_candidate_points(scene_name, offsets)
        if world_points.size == 0:
            return []
        local_points = (world_points - anchor_pos) @ rot.T
        keep = np.linalg.norm(local_points, axis=1) <= self.map_range_m
        return [point.astype(np.float32) for point in local_points[keep]]

    def _world_map_target_candidate_points(
        self,
        scene_name: str,
        lateral_offsets_m: tuple[float, ...],
    ) -> np.ndarray:
        key = (scene_name, lateral_offsets_m)
        if key in self._target_map_point_cache:
            return self._target_map_point_cache[key]

        points: list[np.ndarray] = []
        for map_polyline in self._scene_map_polylines(scene_name):
            if self.target_candidate_type_ids is not None and map_polyline.type_id not in self.target_candidate_type_ids:
                continue
            for start, end in zip(map_polyline.points[:-1], map_polyline.points[1:]):
                vector = end - start
                length = float(np.linalg.norm(vector))
                if length <= 1e-3:
                    continue
                direction = vector / length
                normal = np.array([-direction[1], direction[0]], dtype=np.float32)
                count = max(1, int(np.ceil(length / max(self.target_candidate_spacing_m, 1e-3))))
                for step in range(count + 1):
                    base = start + vector * (step / count)
                    for offset in lateral_offsets_m:
                        point = base + normal * offset
                        points.append(point.astype(np.float32))
        unique = self._dedupe_target_candidates(points)
        if unique:
            world_points = np.stack(unique).astype(np.float32)
        else:
            world_points = np.zeros((0, 2), dtype=np.float32)
        self._target_map_point_cache[key] = world_points
        return world_points

    def _local_grid_target_candidates(self) -> list[np.ndarray]:
        key = (float(self.map_range_m), float(self.target_candidate_grid_spacing_m))
        if key in self._target_grid_cache:
            return self._target_grid_cache[key]

        spacing = max(self.target_candidate_grid_spacing_m, 1e-3)
        coords = np.arange(-self.map_range_m, self.map_range_m + 0.5 * spacing, spacing, dtype=np.float32)
        points: list[np.ndarray] = []
        for x in coords:
            for y in coords:
                point = np.array([x, y], dtype=np.float32)
                if float(np.linalg.norm(point)) <= self.map_range_m:
                    points.append(point)
        self._target_grid_cache[key] = points
        return points

    def _quota_hybrid_target_candidates(
        self,
        scene_name: str,
        anchor_pos: np.ndarray,
        rot: np.ndarray,
        anchor_vel: np.ndarray | None,
    ) -> list[np.ndarray]:
        lane_quota, grid_quota = self._target_candidate_quotas()
        lane_points = self._map_target_candidate_points(
            scene_name,
            anchor_pos,
            rot,
            self._target_candidate_offsets_for_source("quota_hybrid"),
        )
        grid_points = self._local_grid_target_candidates()
        lane_selected = self._select_target_candidates(lane_points, max_count=lane_quota)
        grid_selected = self._select_target_candidates(grid_points, max_count=grid_quota)
        selected = self._dedupe_target_candidates(lane_selected + grid_selected)
        return self._select_target_candidates(selected, max_count=self.max_target_candidates)

    def _reachable_quota_hybrid_target_candidates(
        self,
        scene_name: str,
        anchor_pos: np.ndarray,
        rot: np.ndarray,
        anchor_vel: np.ndarray | None,
    ) -> list[np.ndarray]:
        lane_quota, grid_quota = self._target_candidate_quotas()
        velocity = np.zeros(2, dtype=np.float32) if anchor_vel is None else (rot @ anchor_vel.astype(np.float32))
        horizon_s = float(self.future_steps) * self.target_candidate_reachable_dt_s

        lane_points = self._map_target_candidate_points(
            scene_name,
            anchor_pos,
            rot,
            self._target_candidate_offsets_for_source("reachable_quota_hybrid"),
        )
        grid_points = self._local_grid_target_candidates()
        lane_selected = self._select_reachable_target_candidates(lane_points, lane_quota, velocity, horizon_s)
        grid_selected = self._select_reachable_target_candidates(grid_points, grid_quota, velocity, horizon_s)
        selected = self._dedupe_target_candidates(lane_selected + grid_selected)
        if len(selected) > self.max_target_candidates:
            selected = self._select_target_candidates(selected, max_count=self.max_target_candidates, nearest_fraction=0.0)
        return selected

    def _target_candidate_quotas(self) -> tuple[int, int]:
        lane_quota = self.target_candidate_lane_quota
        grid_quota = self.target_candidate_grid_quota
        if lane_quota is None and grid_quota is None:
            lane_quota = self.max_target_candidates // 2
            grid_quota = self.max_target_candidates - lane_quota
        elif lane_quota is None:
            grid_quota = int(grid_quota or 0)
            lane_quota = max(self.max_target_candidates - grid_quota, 0)
        elif grid_quota is None:
            lane_quota = int(lane_quota)
            grid_quota = max(self.max_target_candidates - lane_quota, 0)
        return int(lane_quota), int(grid_quota)

    def _dedupe_target_candidates(self, points: list[np.ndarray], quant_m: float | None = None) -> list[np.ndarray]:
        unique: dict[tuple[int, int], np.ndarray] = {}
        quant = max(float(quant_m or min(self.target_candidate_spacing_m, self.target_candidate_grid_spacing_m)), 1e-3)
        for point in points:
            key = (int(round(float(point[0]) / quant)), int(round(float(point[1]) / quant)))
            unique.setdefault(key, point.astype(np.float32))
        return list(unique.values())

    def _select_target_candidates(
        self,
        points: list[np.ndarray],
        max_count: int | None = None,
        nearest_fraction: float | None = None,
    ) -> list[np.ndarray]:
        max_count = int(max_count if max_count is not None else self.max_target_candidates)
        if max_count <= 0:
            return []
        if len(points) <= max_count:
            return sorted(points, key=lambda point: float(np.linalg.norm(point)))

        fraction = self.target_candidate_nearest_fraction if nearest_fraction is None else float(nearest_fraction)
        nearest_count = int(round(max_count * fraction))
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
            radial_bin = min(radial_bins - 1, int(distance / max(self.map_range_m, 1e-3) * radial_bins))
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

    def _select_reachable_target_candidates(
        self,
        points: list[np.ndarray],
        max_count: int,
        current_velocity_mps: np.ndarray,
        horizon_s: float,
    ) -> list[np.ndarray]:
        if max_count <= 0:
            return []
        if len(points) <= max_count:
            return sorted(points, key=lambda point: float(np.linalg.norm(point)))

        stacked = np.stack(points).astype(np.float32)
        speed = float(np.linalg.norm(current_velocity_mps))
        cv_endpoint = current_velocity_mps.astype(np.float32) * float(horizon_s)
        dist_to_cv = np.linalg.norm(stacked - cv_endpoint[None, :], axis=1)
        dist_to_origin = np.linalg.norm(stacked, axis=1)
        if speed > 1e-3:
            direction = current_velocity_mps / speed
            longitudinal = stacked @ direction
            lateral_vec = stacked - longitudinal[:, None] * direction[None, :]
            lateral = np.linalg.norm(lateral_vec, axis=1)
            backward_penalty = np.maximum(-longitudinal, 0.0) * 2.0
        else:
            lateral = np.zeros(stacked.shape[0], dtype=np.float32)
            backward_penalty = np.zeros(stacked.shape[0], dtype=np.float32)
        reach_radius = max(15.0, speed * float(horizon_s) + 0.5 * 3.0 * float(horizon_s) ** 2 + 8.0)
        over_reach_penalty = np.maximum(dist_to_origin - reach_radius, 0.0) * 1.5
        score = dist_to_cv + 0.15 * lateral + 0.05 * dist_to_origin + backward_penalty + over_reach_penalty

        primary_count = min(max_count, int(round(max_count * self.target_candidate_reachable_primary_fraction)))
        selected: list[np.ndarray] = []
        selected_ids: set[int] = set()
        if primary_count > 0:
            primary_idx = np.argpartition(score, primary_count - 1)[:primary_count]
            primary_idx = primary_idx[np.argsort(score[primary_idx])]
            selected = [points[int(idx)] for idx in primary_idx]
            selected_ids = {int(idx) for idx in primary_idx}
        remaining = [point for idx, point in enumerate(points) if idx not in selected_ids]
        fill_count = max_count - len(selected)
        if fill_count > 0:
            selected.extend(self._select_target_candidates(remaining, max_count=fill_count, nearest_fraction=0.0))
        return selected[:max_count]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.samples[index]
        track_file = self.files[ref.file_idx]
        track = track_file.tracks[ref.track_id]

        hist_start = ref.end_index - self.history_steps + 1
        hist = track.iloc[hist_start : ref.end_index + 1]
        fut = track.iloc[ref.end_index + 1 : ref.end_index + self.future_steps + 1]
        anchor = track.iloc[ref.end_index]

        anchor_pos = np.array([anchor.x, anchor.y], dtype=np.float32)
        anchor_psi = float(anchor.psi_rad)
        rot = _rotation_to_target_frame(anchor_psi)

        polylines, polyline_mask, segment_mask, polyline_types = self._empty_output()
        self._write_agent_polyline(
            polylines,
            polyline_mask,
            segment_mask,
            polyline_types,
            slot=0,
            rows=[row for _, row in hist.iterrows()],
            anchor_pos=anchor_pos,
            rot=rot,
            type_id=0,
        )

        neighbor_ids = self._selected_neighbor_ids(track_file, ref.track_id, anchor)
        hist_frame_ids = [int(frame_id) for frame_id in hist["frame_id"].tolist()]
        for n_idx, track_id in enumerate(neighbor_ids, start=1):
            rows_by_frame = track_file.track_frames.get(track_id, {})
            rows = [rows_by_frame[frame_id] for frame_id in hist_frame_ids if frame_id in rows_by_frame]
            self._write_agent_polyline(
                polylines,
                polyline_mask,
                segment_mask,
                polyline_types,
                slot=n_idx,
                rows=rows,
                anchor_pos=anchor_pos,
                rot=rot,
                type_id=1,
            )

        scene_name = scene_name_from_track_path(track_file.path)
        self._write_map_polylines(
            polylines,
            polyline_mask,
            segment_mask,
            polyline_types,
            scene_name=scene_name,
            anchor_pos=anchor_pos,
            rot=rot,
        )

        future_xy = fut[["x", "y"]].to_numpy(np.float32)
        target = ((future_xy - anchor_pos) @ rot.T) / self.coordinate_scale_m
        metric_final_state = fut.iloc[-1]
        metric_gt_final_yaw = float(metric_final_state.psi_rad)
        metric_gt_final_speed = float(
            np.hypot(float(metric_final_state.vx), float(metric_final_state.vy))
        )
        anchor_state = np.array(
            [anchor_pos[0], anchor_pos[1], anchor_psi, metric_gt_final_yaw, metric_gt_final_speed],
            dtype=np.float32,
        )
        ref_array = np.array([ref.file_idx, ref.track_id, ref.end_index], dtype=np.int64)
        item = {
            "polylines": torch.from_numpy(polylines),
            "polyline_mask": torch.from_numpy(polyline_mask),
            "segment_mask": torch.from_numpy(segment_mask),
            "polyline_types": torch.from_numpy(polyline_types),
            "target": torch.from_numpy(target.astype(np.float32)),
            "anchor": torch.from_numpy(anchor_state),
            "ref": torch.from_numpy(ref_array),
        }
        if self.max_target_candidates > 0:
            anchor_vel = np.array([float(anchor.vx), float(anchor.vy)], dtype=np.float32)
            target_candidates, candidate_mask = self._target_candidates(scene_name, anchor_pos, rot, anchor_vel)
            item["target_candidates"] = torch.from_numpy(target_candidates)
            item["candidate_mask"] = torch.from_numpy(candidate_mask)
        return item


class CachedPolylineDataset(Dataset):
    """Tensor cache produced by precompute_cache.py."""

    TENSOR_KEYS = ("polylines", "polyline_mask", "segment_mask", "polyline_types", "target", "anchor", "ref")

    def __init__(self, cache_path: str | Path | Sequence[str | Path]) -> None:
        cache_paths = [Path(p) for p in cache_path] if isinstance(cache_path, (list, tuple)) else [Path(cache_path)]
        caches = [self._load_cache(path) for path in cache_paths]
        self._validate_compatible(caches, cache_paths)

        self.cache_paths = cache_paths
        self.cache_path = cache_paths[0] if len(cache_paths) == 1 else cache_paths
        self.meta = caches[0]["meta"]
        refs: list[torch.Tensor] = []
        file_paths: list[str] = []
        scene_names: list[str] = []
        file_offset = 0
        for cache in caches:
            ref = cache["ref"].long().clone()
            ref[:, 0] += file_offset
            refs.append(ref)
            cache_file_paths = [str(path) for path in cache["meta"].get("file_paths", [])]
            cache_scene_names = [str(name) for name in cache["meta"].get("scene_names", [])]
            file_paths.extend(cache_file_paths)
            scene_names.extend(cache_scene_names)
            file_offset += len(cache_file_paths)
        self.polylines = torch.cat([cache["polylines"].float() for cache in caches], dim=0)
        self.polyline_mask = torch.cat([cache["polyline_mask"].bool() for cache in caches], dim=0)
        self.segment_mask = torch.cat([cache["segment_mask"].bool() for cache in caches], dim=0)
        self.polyline_types = torch.cat([cache["polyline_types"].long() for cache in caches], dim=0)
        self.target = torch.cat([cache["target"].float() for cache in caches], dim=0)
        self.anchor = torch.cat([cache["anchor"].float() for cache in caches], dim=0)
        self.ref = torch.cat(refs, dim=0)
        self.target_candidates = (
            torch.cat([cache["target_candidates"].float() for cache in caches], dim=0)
            if "target_candidates" in caches[0]
            else None
        )
        self.candidate_mask = (
            torch.cat([cache["candidate_mask"].bool() for cache in caches], dim=0)
            if "candidate_mask" in caches[0]
            else None
        )
        self.file_paths = file_paths
        self.scene_names = scene_names

        self.feature_dim = int(self.meta["feature_dim"])
        self.future_steps = int(self.meta["future_steps"])
        self.coordinate_scale_m = float(self.meta["coordinate_scale_m"])
        self.target_mode = str(self.meta.get("target_mode", "ego_local"))
        self.max_polylines = int(self.meta["max_polylines"])
        self.max_segments = int(self.meta["max_segments"])
        self.include_map = bool(self.meta.get("include_map", False))
        self.include_velocity = bool(self.meta.get("include_velocity", self.feature_dim >= 12))
        self.sensor_range_m = float(self.meta["sensor_range_m"])
        self.map_range_m = float(self.meta.get("map_range_m", self.sensor_range_m))
        self.max_target_candidates = int(self.meta.get("max_target_candidates", 0))
        self.target_candidate_spacing_m = float(self.meta.get("target_candidate_spacing_m", 0.0))
        self.target_candidate_nearest_fraction = float(self.meta.get("target_candidate_nearest_fraction", 0.5))
        self.target_candidate_source = str(self.meta.get("target_candidate_source", "all"))
        self.target_candidate_type_ids = (
            tuple(int(value) for value in self.meta.get("target_candidate_type_ids", []))
            if self.meta.get("target_candidate_type_ids") is not None
            else None
        )
        self.target_candidate_lateral_offsets_m = tuple(
            float(value) for value in self.meta.get("target_candidate_lateral_offsets_m", [0.0])
        )
        self.target_candidate_grid_spacing_m = float(self.meta.get("target_candidate_grid_spacing_m", 2.0))
        self.target_candidate_lane_quota = self.meta.get("target_candidate_lane_quota")
        self.target_candidate_grid_quota = self.meta.get("target_candidate_grid_quota")
        self.target_candidate_selection = str(
            self.meta.get("target_candidate_selection", "nearest_plus_radial_angular_coverage")
        )
        self.target_candidate_reachable_dt_s = float(self.meta.get("target_candidate_reachable_dt_s", 0.1))
        self.target_candidate_reachable_primary_fraction = float(
            self.meta.get("target_candidate_reachable_primary_fraction", 0.6)
        )

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @classmethod
    def _validate_compatible(cls, caches: list[dict[str, Any]], paths: list[Path]) -> None:
        if not caches:
            raise ValueError("No cache paths were provided.")
        for cache, path in zip(caches, paths):
            missing = [key for key in cls.TENSOR_KEYS + ("meta",) if key not in cache]
            if missing:
                raise KeyError(f"Cache {path} is missing keys: {missing}. Rebuild it with precompute_cache.py.")

        base = caches[0]
        base_meta = base["meta"]
        shape_keys = ("polylines", "polyline_mask", "segment_mask", "polyline_types", "target", "anchor")
        has_candidates = "target_candidates" in base
        for cache, path in zip(caches[1:], paths[1:]):
            if ("target_candidates" in cache) != has_candidates:
                raise ValueError(f"Cache candidate field mismatch in {path}. Rebuild caches with the same config.")
            for key in shape_keys:
                if tuple(cache[key].shape[1:]) != tuple(base[key].shape[1:]):
                    raise ValueError(
                        f"Cache shape mismatch for '{key}' in {path}: "
                        f"expected {tuple(base[key].shape[1:])}, got {tuple(cache[key].shape[1:])}."
                    )
            meta = cache["meta"]
            for key in (
                "history_steps",
                "future_steps",
                "num_neighbors",
                "sensor_range_m",
                "coordinate_scale_m",
                "map_range_m",
                "max_map_polylines",
                "max_segments",
                "include_map",
                "include_velocity",
                "feature_dim",
                "samples_per_scenario",
            ):
                if meta.get(key) != base_meta.get(key):
                    raise ValueError(
                        f"Cache metadata mismatch for '{key}' in {path}: "
                        f"expected {base_meta.get(key)!r}, got {meta.get(key)!r}."
                    )
            if has_candidates:
                for key in TARGET_CANDIDATE_META_KEYS:
                    if meta.get(key) != base_meta.get(key):
                        raise ValueError(
                            f"Cache target-candidate metadata mismatch for '{key}' in {path}: "
                            f"expected {base_meta.get(key)!r}, got {meta.get(key)!r}."
                        )
                for key in ("target_candidates", "candidate_mask"):
                    if tuple(cache[key].shape[1:]) != tuple(base[key].shape[1:]):
                        raise ValueError(
                            f"Cache shape mismatch for '{key}' in {path}: "
                            f"expected {tuple(base[key].shape[1:])}, got {tuple(cache[key].shape[1:])}."
                        )

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "polylines": self.polylines[index],
            "polyline_mask": self.polyline_mask[index],
            "segment_mask": self.segment_mask[index],
            "polyline_types": self.polyline_types[index],
            "target": self.target[index],
            "anchor": self.anchor[index],
            "ref": self.ref[index],
        }
        if self.target_candidates is not None and self.candidate_mask is not None:
            item["target_candidates"] = self.target_candidates[index]
            item["candidate_mask"] = self.candidate_mask[index]
        return item
