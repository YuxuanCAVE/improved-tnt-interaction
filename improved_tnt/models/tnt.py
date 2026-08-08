from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from improved_tnt.models.vectornet import PolylineSubgraphEncoder, StackedGlobalAttentionGraph


class TNTTargetPrediction(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        offset_limit: float = 0.05,
    ) -> None:
        super().__init__()
        self.offset_limit = float(offset_limit)
        self.shared = nn.Sequential(
            nn.Linear(graph_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logit = nn.Linear(hidden_dim, 1)
        self.offset = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(
        self,
        target_feature: torch.Tensor,
        target_candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_candidates = target_candidates.shape[1]
        target_context = target_feature.unsqueeze(1).expand(-1, num_candidates, -1)
        hidden = self.shared(torch.cat([target_context, target_candidates], dim=-1))
        logits = self.logit(hidden).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask.bool(), torch.finfo(logits.dtype).min)
        offsets = torch.tanh(self.offset(hidden)) * self.offset_limit
        return logits, offsets


class TNTMotionEstimation(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        future_steps: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        predict_offsets: bool = False,
        endpoint_exact_residual: bool = False,
    ) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.predict_offsets = bool(predict_offsets)
        self.endpoint_exact_residual = bool(endpoint_exact_residual)
        self.decoder = nn.Sequential(
            nn.Linear(graph_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.future_steps * 2),
        )

    def forward(self, target_feature: torch.Tensor, endpoints: torch.Tensor) -> torch.Tensor:
        if endpoints.ndim == 2:
            endpoints = endpoints.unsqueeze(1)
            squeeze = True
        elif endpoints.ndim == 3:
            squeeze = False
        else:
            raise ValueError(f"Expected endpoints [B,2] or [B,M,2], got {tuple(endpoints.shape)}")

        batch_size, num_targets, _ = endpoints.shape
        target_context = target_feature.unsqueeze(1).expand(-1, num_targets, -1)
        pred = self.decoder(torch.cat([target_context, endpoints], dim=-1))
        pred = pred.view(batch_size, num_targets, self.future_steps, 2)
        if self.predict_offsets:
            pred = torch.cumsum(pred, dim=2)
        if self.endpoint_exact_residual:
            progress = torch.linspace(
                1.0 / self.future_steps,
                1.0,
                self.future_steps,
                device=pred.device,
                dtype=pred.dtype,
            ).view(1, 1, self.future_steps, 1)
            endpoint_correction = endpoints - pred[:, :, -1]
            pred = pred + progress * endpoint_correction.unsqueeze(2)
        return pred[:, 0] if squeeze else pred


class TNTTrajectoryScoring(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        future_steps: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(graph_dim + future_steps * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, target_feature: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        batch_size, num_modes, _, _ = trajectories.shape
        target_context = target_feature.unsqueeze(1).expand(-1, num_modes, -1)
        score_input = torch.cat([target_context, trajectories.reshape(batch_size, num_modes, -1)], dim=-1)
        return self.score(score_input).squeeze(-1)


class TNTVectorNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        future_steps: int,
        subgraph_hidden_dim: int = 64,
        subgraph_layers: int = 3,
        graph_dim: int = 128,
        global_graph_layers: int = 2,
        target_hidden_dim: int = 128,
        motion_hidden_dim: int = 128,
        score_hidden_dim: int = 128,
        tnt_top_m: int = 50,
        tnt_output_k: int = 1,
        target_offset_limit: float = 0.05,
        dropout: float = 0.1,
        predict_offsets: bool = False,
        architecture: str = "paper",
        use_refined_targets: bool = True,
        trajectory_nms_threshold: float = 0.02,
        endpoint_exact_residual: bool = False,
    ) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.tnt_top_m = int(tnt_top_m)
        self.tnt_output_k = int(tnt_output_k)
        self.architecture = architecture
        self.use_paper_architecture = architecture == "paper"
        self.use_refined_targets = bool(use_refined_targets)
        self.trajectory_nms_threshold = float(trajectory_nms_threshold)
        if self.tnt_top_m < 1:
            raise ValueError("tnt_top_m must be >= 1")
        if self.tnt_output_k < 1:
            raise ValueError("tnt_output_k must be >= 1")

        self.traj_subgraph = PolylineSubgraphEncoder(input_dim, subgraph_hidden_dim, subgraph_layers)
        self.map_subgraph = (
            PolylineSubgraphEncoder(input_dim, subgraph_hidden_dim, subgraph_layers)
            if self.use_paper_architecture
            else self.traj_subgraph
        )
        self.global_graph = StackedGlobalAttentionGraph(
            self.traj_subgraph.output_dim,
            graph_dim=graph_dim,
            num_layers=global_graph_layers,
            use_layer_norm=True,
        )
        self.target_prediction = TNTTargetPrediction(
            graph_dim,
            hidden_dim=target_hidden_dim,
            dropout=dropout,
            offset_limit=target_offset_limit,
        )
        self.motion_estimation = TNTMotionEstimation(
            graph_dim,
            future_steps,
            hidden_dim=motion_hidden_dim,
            dropout=dropout,
            predict_offsets=predict_offsets,
            endpoint_exact_residual=endpoint_exact_residual,
        )
        self.trajectory_scoring = TNTTrajectoryScoring(
            graph_dim,
            future_steps,
            hidden_dim=score_hidden_dim,
            dropout=dropout,
        )

    def _encode(
        self,
        polylines: torch.Tensor,
        polyline_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        polyline_types: torch.Tensor,
    ) -> torch.Tensor:
        segment_mask = segment_mask.bool()
        polyline_mask = polyline_mask.bool()
        if self.use_paper_architecture:
            traj_features = self.traj_subgraph(polylines, segment_mask)
            map_features = self.map_subgraph(polylines, segment_mask)
            is_map = polyline_types >= 2
            polyline_features = torch.where(is_map.unsqueeze(-1), map_features, traj_features)
            polyline_features = F.normalize(polyline_features, p=2, dim=-1)
        else:
            polyline_features = self.traj_subgraph(polylines, segment_mask)
        return self.global_graph(polyline_features, polyline_mask)

    def _select_trajectories_with_nms(
        self,
        trajectories: torch.Tensor,
        score_logits: torch.Tensor,
        selected_mask: torch.Tensor,
        selected_k: int,
    ) -> torch.Tensor:
        if selected_k == 1 or self.trajectory_nms_threshold <= 0.0:
            return score_logits.topk(selected_k, dim=1).indices

        batch_size, num_modes, future_steps, coord_dim = trajectories.shape
        sorted_indices = score_logits.argsort(dim=1, descending=True)
        sorted_trajectories = trajectories.gather(
            1,
            sorted_indices[:, :, None, None].expand(batch_size, num_modes, future_steps, coord_dim),
        )
        sorted_valid = selected_mask.bool().gather(1, sorted_indices)
        selected_positions_mask = torch.zeros_like(sorted_valid)
        suppressed = ~sorted_valid
        batch_idx = torch.arange(batch_size, device=trajectories.device)
        threshold_sq = self.trajectory_nms_threshold * self.trajectory_nms_threshold
        selected_positions: list[torch.Tensor] = []
        for _ in range(selected_k):
            nms_available = sorted_valid & ~suppressed & ~selected_positions_mask
            fallback_available = sorted_valid & ~selected_positions_mask
            use_nms = nms_available.any(dim=1)
            available = torch.where(use_nms[:, None], nms_available, fallback_available)
            has_available = available.any(dim=1)
            position = available.float().argmax(dim=1)
            if selected_positions:
                position = torch.where(has_available, position, selected_positions[-1])
            else:
                position = torch.where(has_available, position, torch.zeros_like(position))
            selected_positions.append(position)
            selected_positions_mask[batch_idx, position] = True

            chosen = sorted_trajectories[batch_idx, position]
            distance_sq = (sorted_trajectories - chosen[:, None]).square().sum(dim=-1).max(dim=-1).values
            suppressed = suppressed | (distance_sq <= threshold_sq)

        selected_positions_tensor = torch.stack(selected_positions, dim=1)
        return sorted_indices.gather(1, selected_positions_tensor)

    def forward(
        self,
        polylines: torch.Tensor,
        polyline_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        polyline_types: torch.Tensor | None = None,
        target_candidates: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        target_endpoint: torch.Tensor | None = None,
        output_k: int | None = None,
        return_dict: bool = False,
    ):
        if polyline_types is None:
            raise ValueError("TNTVectorNet requires polyline_types.")
        if target_candidates is None or candidate_mask is None:
            raise ValueError("TNTVectorNet requires target_candidates and candidate_mask.")

        graph_features = self._encode(polylines, polyline_mask, segment_mask, polyline_types)
        target_feature = graph_features[:, 0]
        target_logits, target_offsets = self.target_prediction(target_feature, target_candidates, candidate_mask)
        refined_targets = target_candidates + target_offsets

        top_m = min(self.tnt_top_m, target_candidates.shape[1])
        _, target_indices = target_logits.topk(top_m, dim=1)
        batch_idx = torch.arange(polylines.shape[0], device=polylines.device)
        raw_selected_targets = target_candidates[batch_idx.unsqueeze(1), target_indices]
        refined_selected_targets = refined_targets[batch_idx.unsqueeze(1), target_indices]
        selected_targets = refined_selected_targets if self.use_refined_targets else raw_selected_targets
        selected_mask = candidate_mask.bool()[batch_idx.unsqueeze(1), target_indices]

        candidate_trajectories = self.motion_estimation(target_feature, selected_targets)
        score_logits = self.trajectory_scoring(target_feature, candidate_trajectories)
        score_logits = score_logits.masked_fill(~selected_mask, torch.finfo(score_logits.dtype).min)
        selected_k = min(int(output_k or self.tnt_output_k), candidate_trajectories.shape[1])
        if return_dict:
            score_order = score_logits.topk(selected_k, dim=1).indices
        else:
            score_order = self._select_trajectories_with_nms(
                candidate_trajectories,
                score_logits,
                selected_mask,
                selected_k,
            )
        pred = candidate_trajectories[batch_idx.unsqueeze(1), score_order]
        if selected_k == 1:
            pred = pred[:, 0]

        output = {
            "prediction": pred,
            "target_feature": target_feature,
            "target_logits": target_logits,
            "target_offsets": target_offsets,
            "refined_targets": refined_targets,
            "target_indices": target_indices,
            "raw_selected_targets": raw_selected_targets,
            "selected_targets": selected_targets,
            "selected_mask": selected_mask,
            "candidate_trajectories": candidate_trajectories,
            "trajectory_score_logits": score_logits,
            "score_order": score_order,
        }
        if target_endpoint is not None:
            output["trajectory_with_gt"] = self.motion_estimation(target_feature, target_endpoint)
        return output if return_dict else pred


def build_tnt_model(
    input_dim: int,
    future_steps: int,
    subgraph_hidden_dim: int = 64,
    subgraph_layers: int = 3,
    graph_dim: int = 128,
    global_graph_layers: int = 2,
    target_hidden_dim: int = 128,
    motion_hidden_dim: int = 128,
    score_hidden_dim: int = 128,
    tnt_top_m: int = 50,
    tnt_output_k: int = 1,
    target_offset_limit: float = 0.05,
    dropout: float = 0.1,
    predict_offsets: bool = False,
    architecture: str = "paper",
    use_refined_targets: bool = True,
    trajectory_nms_threshold: float = 0.02,
    endpoint_exact_residual: bool = False,
) -> TNTVectorNet:
    return TNTVectorNet(
        input_dim=input_dim,
        future_steps=future_steps,
        subgraph_hidden_dim=subgraph_hidden_dim,
        subgraph_layers=subgraph_layers,
        graph_dim=graph_dim,
        global_graph_layers=global_graph_layers,
        target_hidden_dim=target_hidden_dim,
        motion_hidden_dim=motion_hidden_dim,
        score_hidden_dim=score_hidden_dim,
        tnt_top_m=tnt_top_m,
        tnt_output_k=tnt_output_k,
        target_offset_limit=target_offset_limit,
        dropout=dropout,
        predict_offsets=predict_offsets,
        architecture=architecture,
        use_refined_targets=use_refined_targets,
        trajectory_nms_threshold=trajectory_nms_threshold,
        endpoint_exact_residual=endpoint_exact_residual,
    )
