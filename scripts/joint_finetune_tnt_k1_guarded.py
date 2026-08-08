from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improved_tnt.engine.official_metrics import official_minade_minfde_mr
from improved_tnt.losses.tnt_loss import TNTLoss
from improved_tnt.models.tnt_refinement import TNTWeightedTrajectoryRefiner
from improved_tnt.utils.geometry import local_to_global_tensor
from val import _build_polyline_checkpoint_model


TENSOR_KEYS = (
    "polylines",
    "polyline_mask",
    "segment_mask",
    "polyline_types",
    "target_candidates",
    "candidate_mask",
    "target",
    "anchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded joint fine-tuning of T3 context encoding and target prediction."
    )
    parser.add_argument("--t3-checkpoint", type=Path, required=True)
    parser.add_argument("--t6-checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--teacher-cache-path", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scorer-lr", type=float, default=1e-6)
    parser.add_argument("--unfreeze-trajectory-scoring", action="store_true")
    parser.add_argument("--k1-temperature", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--k1-trajectory-weight", type=float, default=1.0)
    parser.add_argument("--k1-endpoint-weight", type=float, default=1.0)
    parser.add_argument("--k1-mr-weight", type=float, default=1.0)
    parser.add_argument("--tnt-weight", type=float, default=0.5)
    parser.add_argument("--feature-distill-weight", type=float, default=0.1)
    parser.add_argument("--target-distill-weight", type=float, default=0.1)
    parser.add_argument("--max-k6-mr-degradation", type=float, default=0.002)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--hard-example-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def configure_trainable_modules(
    model: torch.nn.Module,
    unfreeze_trajectory_scoring: bool = False,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.global_graph, model.target_prediction):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if unfreeze_trajectory_scoring:
        for parameter in model.trajectory_scoring.parameters():
            parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def is_guardrail_eligible(k6_mr: float, baseline_k6_mr: float, tolerance: float) -> bool:
    return float(k6_mr) <= float(baseline_k6_mr) + float(tolerance) + 1e-12


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_t6(checkpoint: dict, device: torch.device) -> TNTWeightedTrajectoryRefiner:
    scale_m = float(checkpoint["coordinate_scale_m"])
    model = TNTWeightedTrajectoryRefiner(
        graph_dim=int(checkpoint["graph_dim"]),
        future_steps=int(checkpoint["future_steps"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        trajectory_hidden_dim=int(checkpoint["trajectory_hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        residual_limit=float(checkpoint["residual_limit_m"]) / scale_m,
    ).to(device)
    model.load_state_dict(checkpoint["refiner_state"])
    model.eval()
    # cuDNN requires an RNN to use training execution mode when gradients
    # propagate through its inputs. The frozen one-layer GRU has no dropout.
    model.trajectory_encoder.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_batch(raw: dict, index: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: raw[key][index].to(device) for key in TENSOR_KEYS}


def precompute_teacher_targets(
    teacher: torch.nn.Module,
    raw: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    count = int(raw["target"].shape[0])
    features: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    teacher.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            index = torch.arange(start, end)
            batch = load_batch(raw, index, device)
            graph = teacher._encode(
                batch["polylines"],
                batch["polyline_mask"],
                batch["segment_mask"],
                batch["polyline_types"],
            )
            target_feature = graph[:, 0]
            logits, _ = teacher.target_prediction(
                target_feature,
                batch["target_candidates"],
                batch["candidate_mask"],
            )
            features.append(target_feature.cpu().half())
            probabilities.append(F.softmax(logits, dim=1).cpu().half())
            if start % (batch_size * 100) == 0:
                print(f"Teacher targets: {end}/{count}")
    return {
        "feature": torch.cat(features),
        "target_probability": torch.cat(probabilities),
    }


def official_mr_surrogate_per_sample(
    endpoint_m: torch.Tensor,
    target_endpoint_m: torch.Tensor,
    anchor: torch.Tensor,
    softness_m: float = 0.1,
) -> torch.Tensor:
    delta = endpoint_m - target_endpoint_m
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
    return lateral_violation + longitudinal_violation


def official_miss_mask(
    endpoint_m: torch.Tensor,
    target_endpoint_m: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    delta = endpoint_m - target_endpoint_m
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
    return (lateral.abs() > 1.0) | (longitudinal.abs() > longitudinal_threshold)


def target_distribution_distillation(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    valid = candidate_mask.bool()
    student_log_probability = F.log_softmax(student_logits, dim=1).masked_fill(~valid, 0.0)
    teacher_probability = teacher_probability.masked_fill(~valid, 0.0)
    teacher_probability = teacher_probability / teacher_probability.sum(dim=1, keepdim=True).clamp_min(1e-12)
    teacher_log_probability = teacher_probability.clamp_min(1e-12).log()
    return (
        teacher_probability * (teacher_log_probability - student_log_probability)
    ).sum(dim=1).mean()


def weighted_t6_prediction(
    output: dict[str, torch.Tensor],
    t6: TNTWeightedTrajectoryRefiner,
    temperature: float,
) -> torch.Tensor:
    score_logits = output["trajectory_score_logits"]
    weights = F.softmax(score_logits / max(float(temperature), 1e-6), dim=1)
    weighted = (weights[:, :, None, None] * output["candidate_trajectories"]).sum(dim=1)
    prediction, _ = t6(output["target_feature"], weighted)
    return prediction


def select_k6(model: torch.nn.Module, output: dict[str, torch.Tensor]) -> torch.Tensor:
    trajectories = output["candidate_trajectories"]
    selected_k = min(6, trajectories.shape[1])
    indices = model._select_trajectories_with_nms(
        trajectories,
        output["trajectory_score_logits"],
        output["selected_mask"],
        selected_k,
    )
    batch_index = torch.arange(trajectories.shape[0], device=trajectories.device)
    return trajectories[batch_index.unsqueeze(1), indices]


def evaluate(
    model: torch.nn.Module,
    t6: TNTWeightedTrajectoryRefiner,
    temperature: float,
    raw: dict,
    device: torch.device,
    batch_size: int,
    scale_m: float,
) -> dict[str, float]:
    model.eval()
    count = int(raw["target"].shape[0])
    totals = {"k1_ade": 0.0, "k1_fde": 0.0, "k1_mr": 0.0, "k6_ade": 0.0, "k6_fde": 0.0, "k6_mr": 0.0}
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
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
            k1 = weighted_t6_prediction(output, t6, temperature)
            k6 = select_k6(model, output)
            target_global = local_to_global_tensor(batch["target"], batch["anchor"], scale_m)
            for prefix, prediction in (("k1", k1), ("k6", k6)):
                prediction_global = local_to_global_tensor(prediction, batch["anchor"], scale_m)
                ade, fde, mr = official_minade_minfde_mr(
                    prediction_global,
                    target_global,
                    batch["anchor"][:, 3],
                    batch["anchor"][:, 4],
                )
                totals[f"{prefix}_ade"] += float(ade.sum().item())
                totals[f"{prefix}_fde"] += float(fde.sum().item())
                totals[f"{prefix}_mr"] += float(mr.sum().item())
    return {key: value / count for key, value in totals.items()}


def build_tnt_loss(checkpoint: dict, scale_m: float) -> TNTLoss:
    cfg = checkpoint.get("args", {})
    if not isinstance(cfg, dict):
        cfg = vars(cfg)
    return TNTLoss(
        target_loss_weight=float(cfg.get("target_loss_weight", 0.1)),
        motion_loss_weight=float(cfg.get("motion_loss_weight", 1.0)),
        score_loss_weight=float(cfg.get("score_loss_weight", 0.3)),
        score_temperature=float(cfg.get("score_temperature", 0.5)),
        predicted_motion_loss_weight=float(cfg.get("predicted_motion_loss_weight", 0.5)),
        endpoint_consistency_loss_weight=float(cfg.get("endpoint_consistency_loss_weight", 0.0)),
        coordinate_scale_m=scale_m,
        target_loss_type=str(cfg.get("target_loss_type", "soft")),
        target_soft_label_sigma_m=float(cfg.get("target_soft_label_sigma_m", 1.0)),
        target_soft_label_radius_m=float(cfg.get("target_soft_label_radius_m", 3.0)),
        score_cost_type=str(cfg.get("score_cost_type", "official")),
        score_ade_weight=float(cfg.get("score_ade_weight", 1.0)),
        score_fde_weight=float(cfg.get("score_fde_weight", 2.0)),
        score_miss_weight=float(cfg.get("score_miss_weight", 3.0)),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_raw = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    scale_m = float(train_raw["meta"]["coordinate_scale_m"])

    checkpoint = torch.load(args.t3_checkpoint, map_location="cpu", weights_only=False)
    teacher = _build_polyline_checkpoint_model(checkpoint, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = _build_polyline_checkpoint_model(checkpoint, device)
    trainable_names = configure_trainable_modules(
        student, args.unfreeze_trajectory_scoring
    )
    print(f"Trainable tensors: {len(trainable_names)}")
    trainable_modules = ["global_graph", "target_prediction"]
    if args.unfreeze_trajectory_scoring:
        trainable_modules.append("trajectory_scoring")
    print(f"Trainable modules: {', '.join(trainable_modules)}")

    t6_checkpoint = torch.load(args.t6_checkpoint, map_location="cpu", weights_only=False)
    t6 = build_t6(t6_checkpoint, device)
    temperature = (
        float(args.k1_temperature)
        if args.k1_temperature is not None
        else float(t6_checkpoint["temperature"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_cache_path = (
        args.teacher_cache_path
        if args.teacher_cache_path is not None
        else args.output_dir / "teacher_train_targets.pt"
    )
    if teacher_cache_path.exists():
        teacher_targets = torch.load(
            teacher_cache_path, map_location="cpu", weights_only=False
        )
        if int(teacher_targets["feature"].shape[0]) != int(train_raw["target"].shape[0]):
            raise ValueError("Cached teacher targets do not match the training cache.")
        print(f"Reused teacher targets from {teacher_cache_path}")
    else:
        teacher_targets = precompute_teacher_targets(
            teacher, train_raw, device, args.teacher_batch_size
        )
        torch.save(teacher_targets, teacher_cache_path)
        print(f"Saved teacher targets to {teacher_cache_path}")
    del teacher

    primary_parameters = list(student.global_graph.parameters()) + list(
        student.target_prediction.parameters()
    )
    parameter_groups = [{"params": primary_parameters, "lr": args.lr}]
    if args.unfreeze_trajectory_scoring:
        parameter_groups.append(
            {"params": student.trajectory_scoring.parameters(), "lr": args.scorer_lr}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    tnt_loss_fn = build_tnt_loss(checkpoint, scale_m)
    baseline = evaluate(
        student, t6, temperature, val_raw, device, args.batch_size, scale_m
    )
    guardrail = baseline["k6_mr"] + args.max_k6_mr_degradation
    best_key = (baseline["k1_mr"], baseline["k1_fde"], baseline["k1_ade"])
    best_state = state_dict_cpu(student)
    best_metrics = dict(baseline)
    best_epoch = 0
    rows: list[list[float | int]] = [[0, args.lr, 0.0, *baseline.values(), 1]]
    print(
        "Baseline: "
        f"K1 ADE={baseline['k1_ade']:.6f} FDE={baseline['k1_fde']:.6f} MR={baseline['k1_mr']:.6f}; "
        f"K6 ADE={baseline['k6_ade']:.6f} FDE={baseline['k6_fde']:.6f} MR={baseline['k6_mr']:.6f}; "
        f"K6 MR guardrail={guardrail:.6f}"
    )

    count = int(train_raw["target"].shape[0])
    for epoch in range(1, args.epochs + 1):
        student.eval()
        student.global_graph.train()
        student.target_prediction.train()
        if args.unfreeze_trajectory_scoring:
            student.trajectory_scoring.train()
        permutation = torch.randperm(count)
        total_loss = 0.0
        for start in range(0, count, args.batch_size):
            index = permutation[start : start + args.batch_size]
            batch = load_batch(train_raw, index, device)
            output = student(
                batch["polylines"],
                batch["polyline_mask"],
                batch["segment_mask"],
                batch["polyline_types"],
                target_candidates=batch["target_candidates"],
                candidate_mask=batch["candidate_mask"],
                target_endpoint=batch["target"][:, -1],
                return_dict=True,
            )
            tnt_loss, _ = tnt_loss_fn(
                output,
                batch["target"],
                batch["target_candidates"],
                batch["candidate_mask"],
                final_yaw_local=batch["anchor"][:, 3] - batch["anchor"][:, 2],
                final_speed=batch["anchor"][:, 4],
            )
            prediction = weighted_t6_prediction(output, t6, temperature)
            prediction_m = prediction * scale_m
            target_m = batch["target"] * scale_m
            trajectory_loss_per_sample = F.smooth_l1_loss(
                prediction_m, target_m, reduction="none"
            ).mean(dim=(1, 2))
            endpoint_loss_per_sample = F.smooth_l1_loss(
                prediction_m[:, -1], target_m[:, -1], reduction="none"
            ).mean(dim=1)
            mr_loss_per_sample = official_mr_surrogate_per_sample(
                prediction_m[:, -1], target_m[:, -1], batch["anchor"]
            )
            with torch.no_grad():
                miss = official_miss_mask(
                    prediction_m[:, -1], target_m[:, -1], batch["anchor"]
                )
                sample_weight = 1.0 + args.hard_example_weight * miss.float()
                sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
            trajectory_loss = (sample_weight * trajectory_loss_per_sample).mean()
            endpoint_loss = (sample_weight * endpoint_loss_per_sample).mean()
            mr_loss = (sample_weight * mr_loss_per_sample).mean()
            teacher_feature = teacher_targets["feature"][index].to(
                device=device, dtype=torch.float32
            )
            teacher_probability = teacher_targets["target_probability"][index].to(
                device=device, dtype=torch.float32
            )
            feature_distill = F.mse_loss(output["target_feature"], teacher_feature)
            target_distill = target_distribution_distillation(
                output["target_logits"], teacher_probability, batch["candidate_mask"]
            )
            loss = (
                args.tnt_weight * tnt_loss
                + args.k1_trajectory_weight * trajectory_loss
                + args.k1_endpoint_weight * endpoint_loss
                + args.k1_mr_weight * mr_loss
                + args.feature_distill_weight * feature_distill
                + args.target_distill_weight * target_distill
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in student.parameters() if parameter.requires_grad),
                args.grad_clip_norm,
            )
            optimizer.step()
            total_loss += float(loss.item()) * len(index)
            if start % (args.batch_size * 100) == 0:
                print(f"Epoch {epoch:02d} train: {min(start + len(index), count)}/{count}")

        metrics = evaluate(
            student, t6, temperature, val_raw, device, args.batch_size, scale_m
        )
        eligible = is_guardrail_eligible(
            metrics["k6_mr"], baseline["k6_mr"], args.max_k6_mr_degradation
        )
        key = (metrics["k1_mr"], metrics["k1_fde"], metrics["k1_ade"])
        if eligible and key < best_key:
            best_key = key
            best_state = state_dict_cpu(student)
            best_metrics = dict(metrics)
            best_epoch = epoch
        mean_loss = total_loss / count
        rows.append([epoch, args.lr, mean_loss, *metrics.values(), int(eligible)])
        print(
            f"Epoch {epoch:02d}: loss={mean_loss:.6f}; "
            f"K1 ADE={metrics['k1_ade']:.6f} FDE={metrics['k1_fde']:.6f} MR={metrics['k1_mr']:.6f}; "
            f"K6 ADE={metrics['k6_ade']:.6f} FDE={metrics['k6_fde']:.6f} MR={metrics['k6_mr']:.6f}; "
            f"guardrail={'PASS' if eligible else 'FAIL'}"
        )

    output_checkpoint = dict(checkpoint)
    output_checkpoint.update(
        {
            "model_state": best_state,
            "source_checkpoint": str(args.t3_checkpoint),
            "t6_checkpoint": str(args.t6_checkpoint),
            "best_epoch": best_epoch,
            "best_k1_ade_m": best_key[2],
            "best_k1_fde_m": best_key[1],
            "best_k1_mr": best_key[0],
            "best_metrics": best_metrics,
            "baseline_metrics": baseline,
            "max_k6_mr_degradation": args.max_k6_mr_degradation,
            "trainable_modules": trainable_modules,
            "k1_temperature": temperature,
            "joint_finetune_args": vars(args),
        }
    )
    torch.save(
        output_checkpoint,
        args.output_dir / "best_guarded_joint_t3.pt",
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch", "lr", "train_loss",
                "k1_ade_m", "k1_fde_m", "k1_mr",
                "k6_ade_m", "k6_fde_m", "k6_mr", "k6_guardrail_pass",
            ]
        )
        writer.writerows(rows)
    print(f"Best guarded epoch: {best_epoch}")
    print(f"Saved joint fine-tuning artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
