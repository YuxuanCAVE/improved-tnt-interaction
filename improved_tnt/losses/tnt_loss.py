from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TNTLoss(nn.Module):
    def __init__(
        self,
        target_loss_weight: float = 0.1,
        motion_loss_weight: float = 1.0,
        score_loss_weight: float = 0.1,
        score_temperature: float = 0.01,
        predicted_motion_loss_weight: float = 0.0,
        endpoint_consistency_loss_weight: float = 1.0,
        coordinate_scale_m: float = 1.0,
        target_loss_type: str = "soft",
        target_soft_label_sigma_m: float = 1.0,
        target_soft_label_radius_m: float = 3.0,
    ) -> None:
        super().__init__()
        self.target_loss_weight = float(target_loss_weight)
        self.motion_loss_weight = float(motion_loss_weight)
        self.score_loss_weight = float(score_loss_weight)
        self.score_temperature = max(float(score_temperature), 1e-6)
        self.predicted_motion_loss_weight = float(predicted_motion_loss_weight)
        self.endpoint_consistency_loss_weight = float(endpoint_consistency_loss_weight)
        self.coordinate_scale_m = float(coordinate_scale_m)
        self.target_loss_type = str(target_loss_type).lower()
        self.target_soft_label_sigma_m = max(float(target_soft_label_sigma_m), 1e-6)
        self.target_soft_label_radius_m = max(float(target_soft_label_radius_m), 0.0)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: torch.Tensor,
        target_candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        scale = self.coordinate_scale_m
        target_m = target * scale
        target_candidates_m = target_candidates * scale
        target_endpoint_m = target_m[:, -1]
        candidate_dist = torch.linalg.norm(target_candidates_m - target_endpoint_m.unsqueeze(1), dim=-1)
        candidate_dist = candidate_dist.masked_fill(~candidate_mask.bool(), torch.finfo(candidate_dist.dtype).max)
        target_index = candidate_dist.argmin(dim=1)
        batch_idx = torch.arange(target.shape[0], device=target.device)

        if self.target_loss_type == "hard":
            target_cls_loss = F.cross_entropy(pred["target_logits"], target_index)
        elif self.target_loss_type == "soft":
            soft_target = torch.exp(-0.5 * (candidate_dist / self.target_soft_label_sigma_m).square())
            if self.target_soft_label_radius_m > 0.0:
                soft_target = soft_target.masked_fill(candidate_dist > self.target_soft_label_radius_m, 0.0)
            soft_target = soft_target.masked_fill(~candidate_mask.bool(), 0.0)
            empty_rows = soft_target.sum(dim=1) <= 0.0
            if empty_rows.any():
                soft_target[empty_rows] = F.one_hot(
                    target_index[empty_rows],
                    num_classes=target_candidates.shape[1],
                ).to(dtype=soft_target.dtype)
            soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
            target_log_prob = F.log_softmax(pred["target_logits"], dim=1)
            target_cls_loss = -(soft_target.detach() * target_log_prob).sum(dim=1).mean()
        else:
            raise ValueError("target_loss_type must be 'soft' or 'hard'.")
        offset_target = target_endpoint_m - target_candidates_m[batch_idx, target_index]
        offset_pred = pred["target_offsets"][batch_idx, target_index] * scale
        target_offset_loss = F.smooth_l1_loss(offset_pred, offset_target)
        target_loss = target_cls_loss + target_offset_loss

        trajectory_with_gt = pred.get("trajectory_with_gt")
        if trajectory_with_gt is None:
            raise KeyError("TNTLoss requires pred['trajectory_with_gt']; call model with target_endpoint.")
        trajectory_with_gt_m = trajectory_with_gt * scale
        motion_loss = F.smooth_l1_loss(trajectory_with_gt_m, target_m)

        candidate_trajectories = pred["candidate_trajectories"]
        candidate_trajectories_m = candidate_trajectories * scale
        score_logits = pred["trajectory_score_logits"]
        selected_mask = pred["selected_mask"].bool()
        trajectory_distance = (candidate_trajectories_m - target_m.unsqueeze(1)).square().sum(dim=-1).max(dim=-1).values
        trajectory_distance = trajectory_distance.masked_fill(~selected_mask, torch.finfo(trajectory_distance.dtype).max)
        score_target = F.softmax(-trajectory_distance / self.score_temperature, dim=1).detach()
        score_log_prob = F.log_softmax(score_logits, dim=1)
        score_loss = -(score_target * score_log_prob).sum(dim=1).mean()

        predicted_motion_loss = torch.zeros((), device=target.device, dtype=target.dtype)
        if self.predicted_motion_loss_weight > 0.0:
            best_mode = trajectory_distance.argmin(dim=1)
            predicted_best = candidate_trajectories_m[batch_idx, best_mode]
            predicted_motion_loss = F.smooth_l1_loss(predicted_best, target_m)

        selected_targets = pred["selected_targets"] * scale
        candidate_endpoint_loss = F.smooth_l1_loss(
            candidate_trajectories_m[:, :, -1][selected_mask],
            selected_targets[selected_mask],
        )
        gt_endpoint_loss = F.smooth_l1_loss(trajectory_with_gt_m[:, -1], target_endpoint_m)
        endpoint_consistency_loss = candidate_endpoint_loss + gt_endpoint_loss

        loss = (
            self.target_loss_weight * target_loss
            + self.motion_loss_weight * motion_loss
            + self.score_loss_weight * score_loss
            + self.predicted_motion_loss_weight * predicted_motion_loss
            + self.endpoint_consistency_loss_weight * endpoint_consistency_loss
        )
        parts = {
            "target_cls_loss": float(target_cls_loss.item()),
            "target_offset_loss": float(target_offset_loss.item()),
            "endpoint_loss": float(motion_loss.item()),
            "mode_score_loss": float(score_loss.item()),
            "predicted_motion_loss": float(predicted_motion_loss.item()),
            "endpoint_consistency_loss": float(endpoint_consistency_loss.item()),
            "target_candidate_recall_m": float(candidate_dist[batch_idx, target_index].mean().item()),
        }
        return loss, parts
