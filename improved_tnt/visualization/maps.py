from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def vehicle_polygon(x: float, y: float, psi: float, length: float, width: float) -> np.ndarray:
    corners = np.array(
        [
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float32,
    )
    rot = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def tag_value(element: ET.Element, key: str) -> str | None:
    for tag in element.findall("tag"):
        if tag.get("k") == key:
            return tag.get("v")
    return None


def line_style(way: ET.Element) -> dict | None:
    way_type = tag_value(way, "type")
    subtype = tag_value(way, "subtype")
    if way_type == "curbstone":
        return {"color": "black", "linewidth": 1.0, "zorder": 10}
    if way_type == "line_thin":
        style = {"color": "white", "linewidth": 1.0, "zorder": 10}
        if subtype == "dashed":
            style["dashes"] = [10, 10]
        return style
    if way_type == "line_thick":
        style = {"color": "white", "linewidth": 2.0, "zorder": 10}
        if subtype == "dashed":
            style["dashes"] = [10, 10]
        return style
    if way_type in {"pedestrian_marking", "bike_marking"}:
        return {"color": "white", "linewidth": 1.0, "zorder": 10, "dashes": [5, 10]}
    if way_type == "stop_line":
        return {"color": "white", "linewidth": 3.0, "zorder": 10}
    if way_type == "virtual":
        return {"color": "blue", "linewidth": 1.0, "zorder": 10, "dashes": [2, 5]}
    if way_type in {"road_border", "guard_rail"}:
        return {"color": "black", "linewidth": 1.0, "zorder": 10}
    return None


def draw_osm_xy_map(ax: plt.Axes, map_path: Path) -> bool:
    if not map_path.exists():
        return False
    root = ET.parse(map_path).getroot()
    points: dict[int, tuple[float, float]] = {}
    for node in root.findall("node"):
        points[int(node.get("id"))] = (float(node.get("x")), float(node.get("y")))
    if not points:
        return False

    ax.patch.set_facecolor("lightgrey")
    xy = np.array(list(points.values()), dtype=np.float32)
    ax.set_xlim(float(xy[:, 0].min() - 10.0), float(xy[:, 0].max() + 10.0))
    ax.set_ylim(float(xy[:, 1].min() - 10.0), float(xy[:, 1].max() + 10.0))

    for way in root.findall("way"):
        style = line_style(way)
        if style is None:
            continue
        coords = [points[int(nd.get("ref"))] for nd in way.findall("nd") if int(nd.get("ref")) in points]
        if len(coords) < 2:
            continue
        coords_array = np.array(coords, dtype=np.float32)
        ax.plot(coords_array[:, 0], coords_array[:, 1], **style)
    return True


def draw_map_if_available(ax: plt.Axes, args: argparse.Namespace, scene_name: str | None) -> bool:
    if scene_name is None:
        return False
    map_root = args.map_root or (args.interaction_root / "maps")
    xy_map_path = map_root / f"{scene_name}.osm_xy"
    if draw_osm_xy_map(ax, xy_map_path):
        return True

    map_path = map_root / f"{scene_name}.osm"
    if not map_path.exists():
        return False
    python_root = args.interaction_root / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    try:
        from utils import map_vis_without_lanelet

        previous_ax = plt.gca()
        plt.sca(ax)
        map_vis_without_lanelet.draw_map_without_lanelet(
            str(map_path),
            ax,
            args.lat_origin,
            args.lon_origin,
        )
        plt.sca(previous_ax)
        return True
    except Exception as exc:
        print(f"Warning: could not draw map {map_path}: {exc}")
        return False


def set_local_view(
    ax: plt.Axes,
    hist_xy: np.ndarray,
    fut_xy: np.ndarray,
    pred_xy: np.ndarray,
    current_xy: tuple[float, float],
    sensor_range_m: float,
) -> None:
    all_xy = np.vstack([hist_xy, fut_xy, pred_xy.astype(np.float32)])
    center = np.array(current_xy, dtype=np.float32)
    max_overlay_dist = float(np.linalg.norm(all_xy - center[None, :], axis=1).max()) if len(all_xy) else 0.0
    half_range = max(sensor_range_m + 5.0, max_overlay_dist + 15.0, 25.0)
    ax.set_xlim(float(center[0] - half_range), float(center[0] + half_range))
    ax.set_ylim(float(center[1] - half_range), float(center[1] + half_range))


def set_trackfile_view(ax: plt.Axes, track_file) -> None:
    all_xy = []
    for track in track_file.tracks.values():
        all_xy.append(track[["x", "y"]].to_numpy(np.float32))
    if not all_xy:
        return
    xy = np.vstack(all_xy)
    pad = 20.0
    ax.set_xlim(float(xy[:, 0].min() - pad), float(xy[:, 0].max() + pad))
    ax.set_ylim(float(xy[:, 1].min() - pad), float(xy[:, 1].max() + pad))


def expand_view_to_include_overlay(
    ax: plt.Axes,
    base_xlim: tuple[float, float],
    base_ylim: tuple[float, float],
    hist_xy: np.ndarray,
    fut_xy: np.ndarray,
    pred_xy: np.ndarray,
    current_xy: tuple[float, float],
    sensor_range_m: float,
) -> None:
    all_xy = np.vstack([hist_xy, fut_xy, pred_xy.astype(np.float32)])
    x_min = min(float(base_xlim[0]), float(all_xy[:, 0].min()), current_xy[0] - sensor_range_m)
    x_max = max(float(base_xlim[1]), float(all_xy[:, 0].max()), current_xy[0] + sensor_range_m)
    y_min = min(float(base_ylim[0]), float(all_xy[:, 1].min()), current_xy[1] - sensor_range_m)
    y_max = max(float(base_ylim[1]), float(all_xy[:, 1].max()), current_xy[1] + sensor_range_m)
    pad = 5.0
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
