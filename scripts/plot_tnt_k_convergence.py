from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    "minade": "#49c2d9",
    "minfde": "#c85e62",
    "mr": "#67a583",
    "challenge": "#7b95c6",
    "grid": "#d0e2c0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TNT output-budget convergence from ranked K-sweep metrics."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No metric rows found in {path}.")

    values = {
        "k": [int(row["k"]) for row in rows],
        "cases": [int(row["cases"]) for row in rows],
        "minade": [float(row["minade_m"]) for row in rows],
        "minfde": [float(row["minfde_m"]) for row in rows],
        "mr": [float(row["mr"]) for row in rows],
    }

    expected_k = list(range(1, max(values["k"]) + 1))
    if values["k"] != expected_k:
        raise ValueError("K values must be consecutive and start at 1.")
    if len(set(values["cases"])) != 1:
        raise ValueError("Every K value must use the same validation cases.")
    for name in ("minade", "minfde", "mr"):
        if any(
            later > earlier + 1e-9
            for earlier, later in zip(values[name], values[name][1:])
        ):
            raise ValueError(f"{name} is not monotonically non-increasing.")
    return values


def main() -> None:
    args = parse_args()
    values = load_metrics(args.metrics)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    figure, error_axis = plt.subplots(figsize=(7.2, 4.65), constrained_layout=True)
    mr_axis = error_axis.twinx()
    marker_indices = [0, 1, 2, 5, 9, 14, 19, 29, 39, 49]

    ade_line = error_axis.plot(
        values["k"],
        values["minade"],
        color=PALETTE["minade"],
        linewidth=2.3,
        marker="o",
        markersize=4.8,
        markevery=marker_indices,
        label="minADE",
    )[0]
    fde_line = error_axis.plot(
        values["k"],
        values["minfde"],
        color=PALETTE["minfde"],
        linewidth=2.3,
        marker="s",
        markersize=4.8,
        markevery=marker_indices,
        label="minFDE",
    )[0]
    mr_line = mr_axis.plot(
        values["k"],
        values["mr"],
        color=PALETTE["mr"],
        linewidth=2.3,
        marker="^",
        markersize=5.2,
        markevery=marker_indices,
        label="MR",
    )[0]

    challenge_line = error_axis.axvline(
        6,
        color=PALETTE["challenge"],
        linestyle="--",
        linewidth=1.8,
        label="Challenge limit ($K=6$)",
    )

    error_axis.set_xlabel("Number of predicted trajectories, $K$")
    error_axis.set_ylabel("Displacement error (m)")
    mr_axis.set_ylabel("Miss rate")
    error_axis.set_xlim(1, 50)
    error_axis.set_xticks([1, 6, 10, 20, 30, 40, 50])
    error_axis.set_ylim(bottom=0)
    mr_axis.set_ylim(bottom=0)
    error_axis.grid(axis="both", color=PALETTE["grid"], linewidth=0.8, alpha=0.7)
    error_axis.set_axisbelow(True)

    error_axis.legend(
        [ade_line, fde_line, mr_line, challenge_line],
        ["minADE", "minFDE", "MR", "Challenge limit ($K=6$)"],
        loc="upper right",
        frameon=True,
        ncol=2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
