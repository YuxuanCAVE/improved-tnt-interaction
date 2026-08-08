from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_finetune_tnt_k1_guarded import build_t6, load_batch
from improved_tnt.engine.official_metrics import official_minade_minfde_mr
from improved_tnt.utils.geometry import local_to_global_tensor
from val import _build_polyline_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final guarded TNT model from a precomputed validation cache. "
            "The script reports deterministic T3+T6 K=1, ranked/NMS K-sweep, and "
            "scenario-type metrics in one model pass."
        )
    )
    parser.add_argument("--t3-checkpoint", type=Path, required=True)
    parser.add_argument("--t6-checkpoint", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 6, 9, 12, 15])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def scenario_type(scene_name: str) -> str:
    lowered = scene_name.lower()
    if "merging" in lowered:
        return "Merging"
    if "intersection" in lowered:
        return "Intersection"
    if "roundabout" in lowered:
        return "Roundabout"
    return scene_name


def scenario_labels(raw: dict, count: int) -> list[str]:
    scene_names = raw.get("meta", {}).get("scene_names")
    if not scene_names:
        raise ValueError("Validation cache metadata does not contain scene_names.")
    file_indices = raw["ref"][:count, 0].long().tolist()
    if file_indices and max(file_indices) >= len(scene_names):
        raise ValueError(
            f"Cache ref contains file index {max(file_indices)}, but only "
            f"{len(scene_names)} scene_names are available."
        )
    return [scenario_type(str(scene_names[file_index])) for file_index in file_indices]


def empty_totals() -> dict[str, float]:
    return {"ade": 0.0, "fde": 0.0, "mr": 0.0, "count": 0}


def accumulate(
    totals: dict[str, float],
    ade: torch.Tensor,
    fde: torch.Tensor,
    mr: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    if mask is not None:
        ade = ade[mask]
        fde = fde[mask]
        mr = mr[mask]
    totals["ade"] += float(ade.sum().item())
    totals["fde"] += float(fde.sum().item())
    totals["mr"] += float(mr.sum().item())
    totals["count"] += int(ade.numel())


def averaged(totals: dict[str, float]) -> dict[str, float | int]:
    count = int(totals["count"])
    if count == 0:
        return {"cases": 0, "ade": float("nan"), "fde": float("nan"), "mr": float("nan")}
    return {
        "cases": count,
        "ade": totals["ade"] / count,
        "fde": totals["fde"] / count,
        "mr": totals["mr"] / count,
    }


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive.")
    k_values = sorted(set(int(value) for value in args.k_values))
    if not k_values or k_values[0] < 1:
        raise ValueError("--k-values must contain positive integers.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.t3_checkpoint, map_location="cpu", weights_only=False)
    model = _build_polyline_checkpoint_model(checkpoint, device)
    model.eval()

    t6_checkpoint = torch.load(args.t6_checkpoint, map_location="cpu", weights_only=False)
    t6 = build_t6(t6_checkpoint, device)

    raw = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    available_count = int(raw["target"].shape[0])
    count = min(available_count, args.max_cases) if args.max_cases else available_count
    labels = scenario_labels(raw, count)
    scenario_names = sorted(set(labels))
    scale_m = float(raw["meta"]["coordinate_scale_m"])

    max_k = min(k_values[-1], int(model.tnt_top_m))
    if max_k != k_values[-1]:
        raise ValueError(f"Requested K={k_values[-1]} exceeds model top-M={model.tnt_top_m}.")

    deterministic_totals = empty_totals()
    sweep_totals = {k: empty_totals() for k in k_values}
    scenario_totals = {
        name: {"deterministic_k1": empty_totals(), "challenge_k6": empty_totals()}
        for name in scenario_names
    }

    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, count, args.batch_size):
            end = min(start + args.batch_size, count)
            batch = load_batch(raw, torch.arange(start, end), device)
            output = model(
                batch["polylines"],
                batch["polyline_mask"],
                batch["segment_mask"],
                batch["polyline_types"],
                target_candidates=batch["target_candidates"],
                candidate_mask=batch["candidate_mask"],
                return_dict=True,
            )

            logits = output["trajectory_score_logits"]
            weights = F.softmax(logits / args.temperature, dim=1)
            weighted = (
                weights[:, :, None, None] * output["candidate_trajectories"]
            ).sum(dim=1)
            deterministic_k1, _ = t6(output["target_feature"], weighted)

            selected_indices = model._select_trajectories_with_nms(
                output["candidate_trajectories"],
                logits,
                output["selected_mask"],
                max_k,
            )
            batch_indices = torch.arange(end - start, device=device)
            ranked_max = output["candidate_trajectories"][
                batch_indices.unsqueeze(1), selected_indices
            ]

            target_global = local_to_global_tensor(batch["target"], batch["anchor"], scale_m)
            deterministic_global = local_to_global_tensor(
                deterministic_k1, batch["anchor"], scale_m
            )
            k1_ade, k1_fde, k1_mr = official_minade_minfde_mr(
                deterministic_global,
                target_global,
                batch["anchor"][:, 3],
                batch["anchor"][:, 4],
            )
            accumulate(deterministic_totals, k1_ade, k1_fde, k1_mr)

            sweep_batch_metrics: dict[
                int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ] = {}
            for k in k_values:
                prediction_global = local_to_global_tensor(
                    ranked_max[:, :k], batch["anchor"], scale_m
                )
                metrics = official_minade_minfde_mr(
                    prediction_global,
                    target_global,
                    batch["anchor"][:, 3],
                    batch["anchor"][:, 4],
                )
                sweep_batch_metrics[k] = metrics
                accumulate(sweep_totals[k], *metrics)

            if 6 not in sweep_batch_metrics:
                raise ValueError("Scenario reporting requires K=6 in --k-values.")
            k6_ade, k6_fde, k6_mr = sweep_batch_metrics[6]
            batch_labels = labels[start:end]
            for name in scenario_names:
                mask = torch.tensor(
                    [label == name for label in batch_labels],
                    dtype=torch.bool,
                    device=device,
                )
                accumulate(
                    scenario_totals[name]["deterministic_k1"],
                    k1_ade,
                    k1_fde,
                    k1_mr,
                    mask,
                )
                accumulate(
                    scenario_totals[name]["challenge_k6"],
                    k6_ade,
                    k6_fde,
                    k6_mr,
                    mask,
                )

            if start == 0 or end == count or start % (args.batch_size * 100) == 0:
                print(f"Validated: {end}/{count}")

    deterministic = averaged(deterministic_totals)
    sweep = {k: averaged(sweep_totals[k]) for k in k_values}
    scenarios = {
        name: {
            output_name: averaged(values)
            for output_name, values in scenario_totals[name].items()
        }
        for name in scenario_names
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "final_deterministic_k1.csv",
        ["configuration", "cases", "ade_m", "fde_m", "mr"],
        [[
            f"guarded_t3_weighted_t6_temperature_{args.temperature:g}",
            deterministic["cases"],
            deterministic["ade"],
            deterministic["fde"],
            deterministic["mr"],
        ]],
    )
    write_csv(
        args.output_dir / "final_ranked_k_sweep.csv",
        ["k", "cases", "minade_m", "minfde_m", "mr"],
        [
            [k, sweep[k]["cases"], sweep[k]["ade"], sweep[k]["fde"], sweep[k]["mr"]]
            for k in k_values
        ],
    )
    write_csv(
        args.output_dir / "final_scenario_metrics.csv",
        [
            "scenario_type",
            "cases",
            "k1_ade_m",
            "k1_fde_m",
            "k1_mr",
            "k6_minade_m",
            "k6_minfde_m",
            "k6_mr",
        ],
        [
            [
                name,
                scenarios[name]["challenge_k6"]["cases"],
                scenarios[name]["deterministic_k1"]["ade"],
                scenarios[name]["deterministic_k1"]["fde"],
                scenarios[name]["deterministic_k1"]["mr"],
                scenarios[name]["challenge_k6"]["ade"],
                scenarios[name]["challenge_k6"]["fde"],
                scenarios[name]["challenge_k6"]["mr"],
            ]
            for name in scenario_names
        ],
    )

    metadata = {
        "t3_checkpoint": str(args.t3_checkpoint),
        "t6_checkpoint": str(args.t6_checkpoint),
        "validation_cache": str(args.val_cache),
        "precomputed_cache_used": True,
        "available_cases": available_count,
        "evaluated_cases": count,
        "temperature": args.temperature,
        "k_values": k_values,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "deterministic_k1": deterministic,
        "ranked_k_sweep": {str(k): sweep[k] for k in k_values},
        "scenario_metrics": scenarios,
    }
    with (args.output_dir / "final_results_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print(
        "Final deterministic K=1: "
        f"ADE={deterministic['ade']:.6f} "
        f"FDE={deterministic['fde']:.6f} MR={deterministic['mr']:.6f}"
    )
    challenge = sweep[6]
    print(
        "Final challenge K=6: "
        f"minADE={challenge['ade']:.6f} "
        f"minFDE={challenge['fde']:.6f} MR={challenge['mr']:.6f}"
    )
    print(f"Saved final result files to {args.output_dir}")


if __name__ == "__main__":
    main()
