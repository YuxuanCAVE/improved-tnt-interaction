from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improved_tnt.engine.official_metrics import official_minade_minfde_mr
from improved_tnt.models.tnt_refinement import TNTWeightedTrajectoryRefiner
from improved_tnt.utils.geometry import local_to_global_tensor
from val import _build_polyline_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train T6 weighted-trajectory refinement on frozen T3.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--precompute-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--trajectory-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-limit-m", type=float, default=2.0)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--mr-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def precompute(
    model: torch.nn.Module,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> dict[str, torch.Tensor]:
    raw = torch.load(cache_path, map_location="cpu", weights_only=False)
    count = int(raw["target"].shape[0])
    features: list[torch.Tensor] = []
    bases: list[torch.Tensor] = []
    model.eval()
    tensor_keys = (
        "polylines",
        "polyline_mask",
        "segment_mask",
        "polyline_types",
        "target_candidates",
        "candidate_mask",
    )
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            batch = {key: raw[key][start:end].to(device) for key in tensor_keys}
            output = model(
                batch["polylines"],
                batch["polyline_mask"],
                batch["segment_mask"],
                batch["polyline_types"],
                target_candidates=batch["target_candidates"],
                candidate_mask=batch["candidate_mask"],
                return_dict=True,
            )
            weights = F.softmax(output["trajectory_score_logits"] / max(temperature, 1e-6), dim=1)
            weighted = (weights[:, :, None, None] * output["candidate_trajectories"]).sum(dim=1)
            features.append(output["target_feature"].cpu().half())
            bases.append(weighted.cpu().half())
            if start % (batch_size * 100) == 0:
                print(f"Precomputed {cache_path.name}: {end}/{count}")
    result = {
        "features": torch.cat(features),
        "base": torch.cat(bases),
        "target": raw["target"],
        "anchor": raw["anchor"],
        "scale_m": torch.tensor(float(raw["meta"]["coordinate_scale_m"])),
    }
    del raw
    return result


def official_mr_surrogate(
    prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    anchor: torch.Tensor,
    softness_m: float = 0.1,
) -> torch.Tensor:
    delta = prediction_m[:, -1] - target_m[:, -1]
    final_yaw_local = anchor[:, 3] - anchor[:, 2]
    c = torch.cos(final_yaw_local)
    s = torch.sin(final_yaw_local)
    longitudinal = delta[:, 0] * c + delta[:, 1] * s
    lateral = -delta[:, 0] * s + delta[:, 1] * c
    speed = anchor[:, 4].clamp_min(0.0)
    longitudinal_threshold = torch.where(
        speed < 1.4,
        torch.ones_like(speed),
        torch.where(
            speed > 11.0,
            torch.full_like(speed, 2.0),
            1.0 + (speed - 1.4) / (11.0 - 1.4),
        ),
    )
    softness = max(float(softness_m), 1e-6)
    lateral_violation = F.softplus((lateral.abs() - 1.0) / softness) * softness
    longitudinal_violation = F.softplus(
        (longitudinal.abs() - longitudinal_threshold) / softness
    ) * softness
    return (lateral_violation + longitudinal_violation).mean()


def evaluate(
    refiner: TNTWeightedTrajectoryRefiner,
    data: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, float]:
    refiner.eval()
    count = int(data["features"].shape[0])
    scale_m = float(data["scale_m"].item())
    ade_sum = fde_sum = mr_sum = 0.0
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            feature = data["features"][start:end].to(device=device, dtype=torch.float32)
            base = data["base"][start:end].to(device=device, dtype=torch.float32)
            target = data["target"][start:end].to(device)
            anchor = data["anchor"][start:end].to(device)
            prediction, _ = refiner(feature, base)
            prediction_global = local_to_global_tensor(prediction, anchor, scale_m)
            target_global = local_to_global_tensor(target, anchor, scale_m)
            ade, fde, mr = official_minade_minfde_mr(
                prediction_global,
                target_global,
                anchor[:, 3],
                anchor[:, 4],
            )
            ade_sum += float(ade.sum().item())
            fde_sum += float(fde.sum().item())
            mr_sum += float(mr.sum().item())
    return ade_sum / count, fde_sum / count, mr_sum / count


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_polyline_checkpoint_model(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_data = precompute(
        model, args.train_cache, device, args.precompute_batch_size, args.temperature
    )
    val_data = precompute(
        model, args.val_cache, device, args.precompute_batch_size, args.temperature
    )
    scale_m = float(train_data["scale_m"].item())
    graph_dim = int(train_data["features"].shape[1])
    future_steps = int(train_data["target"].shape[1])
    refiner = TNTWeightedTrajectoryRefiner(
        graph_dim=graph_dim,
        future_steps=future_steps,
        hidden_dim=args.hidden_dim,
        trajectory_hidden_dim=args.trajectory_hidden_dim,
        dropout=args.dropout,
        residual_limit=args.residual_limit_m / scale_m,
    ).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-5
    )

    count = int(train_data["features"].shape[0])
    baseline = evaluate(refiner, val_data, device, args.batch_size)
    best_key = (baseline[2], baseline[1], baseline[0])
    best_state = copy.deepcopy(refiner.state_dict())
    best_epoch = 0
    rows = [[0, args.lr, 0.0, *baseline]]
    print(f"Baseline: ADE={baseline[0]:.6f} FDE={baseline[1]:.6f} MR={baseline[2]:.6f}")

    for epoch in range(1, args.epochs + 1):
        refiner.train()
        permutation = torch.randperm(count)
        total_loss = 0.0
        for start in range(0, count, args.batch_size):
            index = permutation[start : start + args.batch_size]
            feature = train_data["features"][index].to(device=device, dtype=torch.float32)
            base = train_data["base"][index].to(device=device, dtype=torch.float32)
            target = train_data["target"][index].to(device)
            anchor = train_data["anchor"][index].to(device)
            prediction, residual = refiner(feature, base)
            prediction_m = prediction * scale_m
            target_m = target * scale_m
            trajectory_loss = F.smooth_l1_loss(prediction_m, target_m)
            endpoint_loss = F.smooth_l1_loss(prediction_m[:, -1], target_m[:, -1])
            mr_loss = official_mr_surrogate(prediction_m, target_m, anchor)
            second_difference = residual[:, 2:] - 2.0 * residual[:, 1:-1] + residual[:, :-2]
            smoothness_loss = second_difference.square().mean() * scale_m * scale_m
            residual_loss = residual.square().mean() * scale_m * scale_m
            loss = (
                trajectory_loss
                + args.endpoint_weight * endpoint_loss
                + args.mr_weight * mr_loss
                + args.smoothness_weight * smoothness_loss
                + args.residual_weight * residual_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(index)

        metrics = evaluate(refiner, val_data, device, args.batch_size)
        scheduler.step(metrics[2])
        lr = optimizer.param_groups[0]["lr"]
        mean_loss = total_loss / count
        rows.append([epoch, lr, mean_loss, *metrics])
        print(
            f"Epoch {epoch:02d}: loss={mean_loss:.6f} lr={lr:.6g} "
            f"ADE={metrics[0]:.6f} FDE={metrics[1]:.6f} MR={metrics[2]:.6f}"
        )
        key = (metrics[2], metrics[1], metrics[0])
        if key < best_key:
            best_key = key
            best_state = copy.deepcopy(refiner.state_dict())
            best_epoch = epoch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    refiner.load_state_dict(best_state)
    torch.save(
        {
            "refiner_state": refiner.state_dict(),
            "source_checkpoint": str(args.checkpoint),
            "temperature": args.temperature,
            "graph_dim": graph_dim,
            "future_steps": future_steps,
            "hidden_dim": args.hidden_dim,
            "trajectory_hidden_dim": args.trajectory_hidden_dim,
            "dropout": args.dropout,
            "residual_limit_m": args.residual_limit_m,
            "coordinate_scale_m": scale_m,
            "best_epoch": best_epoch,
            "best_ade_m": best_key[2],
            "best_fde_m": best_key[1],
            "best_mr": best_key[0],
        },
        args.output_dir / "best_t6_refiner.pt",
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "lr", "train_loss", "ade_m", "fde_m", "mr"])
        writer.writerows(rows)
    print(
        f"Best epoch {best_epoch}: ADE={best_key[2]:.6f} "
        f"FDE={best_key[1]:.6f} MR={best_key[0]:.6f}"
    )
    print(f"Saved T6 artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
