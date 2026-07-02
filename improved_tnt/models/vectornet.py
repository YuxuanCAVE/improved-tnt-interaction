from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    pooled = masked.max(dim=dim).values
    valid = mask.any(dim=dim).unsqueeze(-1)
    return torch.where(valid, pooled, torch.zeros_like(pooled))


class PolylineSubgraphLayer(nn.Module):
    """One VectorNet local subgraph layer for vector segments inside a polyline."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        pooled = _masked_max(encoded, segment_mask, dim=2)
        pooled = pooled.unsqueeze(2).expand(-1, -1, encoded.shape[2], -1)
        out = torch.cat([encoded, pooled], dim=-1)
        return out * segment_mask.unsqueeze(-1)


class PolylineSubgraphEncoder(nn.Module):
    """Encodes padded vector segments into one feature per polyline."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(PolylineSubgraphLayer(current_dim, hidden_dim))
            current_dim = hidden_dim * 2
        self.layers = nn.ModuleList(layers)
        self.output_dim = current_dim

    def forward(self, polylines: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
        x = polylines
        for layer in self.layers:
            x = layer(x, segment_mask)
        return _masked_max(x, segment_mask, dim=2)


class GlobalAttentionGraph(nn.Module):
    """Fully connected self-attention over polyline-level features."""

    def __init__(self, input_dim: int, graph_dim: int = 128, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, graph_dim)
        self.key = nn.Linear(input_dim, graph_dim)
        self.value = nn.Linear(input_dim, graph_dim)
        self.residual = nn.Linear(input_dim, graph_dim) if input_dim != graph_dim else nn.Identity()
        self.norm = nn.LayerNorm(graph_dim) if use_layer_norm else nn.Identity()
        self.scale = graph_dim**0.5

    def forward(self, polyline_features: torch.Tensor, polyline_mask: torch.Tensor) -> torch.Tensor:
        q = F.relu(self.query(polyline_features))
        k = F.relu(self.key(polyline_features))
        v = F.relu(self.value(polyline_features))
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        scores = scores.masked_fill(~polyline_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * polyline_mask.unsqueeze(1)
        context = torch.matmul(attention, v)
        out = self.norm(context + self.residual(polyline_features))
        return out * polyline_mask.unsqueeze(-1)


class StackedGlobalAttentionGraph(nn.Module):
    """Stacked fully connected attention blocks over polyline-level features."""

    def __init__(
        self,
        input_dim: int,
        graph_dim: int = 128,
        num_layers: int = 1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("global_graph_layers must be >= 1")
        self.layers = nn.ModuleList(
            [
                GlobalAttentionGraph(
                    input_dim=input_dim if layer_idx == 0 else graph_dim,
                    graph_dim=graph_dim,
                    use_layer_norm=use_layer_norm,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(self, polyline_features: torch.Tensor, polyline_mask: torch.Tensor) -> torch.Tensor:
        x = polyline_features
        for layer in self.layers:
            x = layer(x, polyline_mask)
        return x


class VectorNetTrajectoryPredictor(nn.Module):
    """VectorNet-style encoder with an MLP trajectory decoder.

    The model expects padded INTERACTION polyline tensors:
        polylines: [B, P, S, F]
        polyline_mask: [B, P]
        segment_mask: [B, P, S]

    Polyline slot 0 must be the target vehicle history. The decoder predicts
    ego-local future positions. With predict_offsets=True, the MLP outputs
    per-step offsets and cumulative sums them along the future horizon, matching
    the parameterization described in VectorNet.
    """

    def __init__(
        self,
        input_dim: int = 8,
        future_steps: int = 30,
        subgraph_hidden_dim: int = 64,
        subgraph_layers: int = 3,
        graph_dim: int = 128,
        global_graph_layers: int = 1,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.1,
        predict_offsets: bool = True,
        architecture: str = "current",
        auxiliary_node_loss: bool = False,
        aux_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.predict_offsets = predict_offsets
        self.architecture = architecture
        self.use_paper_architecture = architecture == "paper"
        self.auxiliary_node_loss = auxiliary_node_loss
        self.traj_subgraph = PolylineSubgraphEncoder(
            input_dim=input_dim,
            hidden_dim=subgraph_hidden_dim,
            num_layers=subgraph_layers,
        )
        if self.use_paper_architecture:
            self.map_subgraph = PolylineSubgraphEncoder(
                input_dim=input_dim,
                hidden_dim=subgraph_hidden_dim,
                num_layers=subgraph_layers,
            )
        else:
            self.map_subgraph = self.traj_subgraph
        self.global_graph = StackedGlobalAttentionGraph(
            self.traj_subgraph.output_dim,
            graph_dim=graph_dim,
            num_layers=global_graph_layers,
            use_layer_norm=not self.use_paper_architecture,
        )
        self.decoder = nn.Sequential(
            nn.Linear(graph_dim, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, future_steps * 2),
        )
        if auxiliary_node_loss:
            self.aux_head = nn.Sequential(
                nn.Linear(graph_dim + 3, aux_hidden_dim),
                nn.LayerNorm(aux_hidden_dim),
                nn.ReLU(),
                nn.Linear(aux_hidden_dim, 4),
            )
        else:
            self.aux_head = None

    def _encode(
        self,
        polylines: torch.Tensor,
        polyline_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        polyline_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        segment_mask = segment_mask.bool()
        polyline_mask = polyline_mask.bool()
        if self.use_paper_architecture and polyline_types is not None:
            # Paper-style VectorNet uses separate local subgraphs for agent
            # trajectories and map polylines before the global interaction graph.
            traj_features = self.traj_subgraph(polylines, segment_mask)
            map_features = self.map_subgraph(polylines, segment_mask)
            is_map = polyline_types >= 2
            polyline_features = torch.where(is_map.unsqueeze(-1), map_features, traj_features)
            polyline_features = F.normalize(polyline_features, p=2, dim=-1)
        else:
            polyline_features = self.traj_subgraph(polylines, segment_mask)
        return self.global_graph(polyline_features, polyline_mask)

    def _apply_auxiliary_mask(
        self,
        polylines: torch.Tensor,
        segment_mask: torch.Tensor,
        aux_mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = segment_mask.bool()
        random_mask = torch.rand(valid.shape, device=valid.device) < aux_mask_ratio
        aux_mask = valid & random_mask
        if not aux_mask.any():
            valid_indices = valid.nonzero(as_tuple=False)
            if valid_indices.numel() == 0:
                return polylines, aux_mask
            chosen = valid_indices[torch.randint(valid_indices.shape[0], (1,), device=valid.device)]
            aux_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
        masked_polylines = polylines.clone()
        masked_polylines[aux_mask, :6] = 0.0
        masked_polylines[aux_mask, 7] = 0.0
        return masked_polylines, aux_mask

    def _auxiliary_completion(
        self,
        graph_features: torch.Tensor,
        polylines: torch.Tensor,
        polyline_types: torch.Tensor,
        aux_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.aux_head is None:
            raise RuntimeError("auxiliary_node_loss is disabled for this model.")
        batch_idx, poly_idx, seg_idx = aux_mask.nonzero(as_tuple=True)
        context = graph_features[batch_idx, poly_idx]
        segment_pos = seg_idx.float().unsqueeze(-1) / max(polylines.shape[2] - 1, 1)
        type_feature = polyline_types[batch_idx, poly_idx].float().unsqueeze(-1) / 10.0
        valid_feature = torch.ones_like(segment_pos)
        aux_input = torch.cat([context, segment_pos, type_feature, valid_feature], dim=-1)
        pred = self.aux_head(aux_input)
        target = polylines[batch_idx, poly_idx, seg_idx, :4]
        return pred, target

    def forward(
        self,
        polylines: torch.Tensor,
        polyline_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        polyline_types: torch.Tensor | None = None,
        return_aux: bool = False,
        aux_mask_ratio: float = 0.15,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux_data: dict[str, torch.Tensor] = {}
        model_input = polylines
        aux_mask = None
        if return_aux and self.auxiliary_node_loss:
            if polyline_types is None:
                raise ValueError("polyline_types is required for auxiliary node completion.")
            model_input, aux_mask = self._apply_auxiliary_mask(polylines, segment_mask, aux_mask_ratio)
        graph_features = self._encode(model_input, polyline_mask, segment_mask, polyline_types)
        target_feature = graph_features[:, 0]
        pred = self.decoder(target_feature).view(polylines.shape[0], self.future_steps, 2)
        if self.predict_offsets:
            pred = torch.cumsum(pred, dim=1)
        if return_aux and self.auxiliary_node_loss and aux_mask is not None and polyline_types is not None:
            aux_pred, aux_target = self._auxiliary_completion(graph_features, polylines, polyline_types, aux_mask)
            aux_data = {"pred": aux_pred, "target": aux_target}
            return pred, aux_data
        return pred


class CandidateTNTVectorNetTrajectoryPredictor(nn.Module):
    """TNT-style model with map-sampled target candidates.

    This keeps the VectorNet encoder, predicts a distribution over discrete
    target candidates plus continuous offsets, decodes trajectories for the
    top-M refined targets, then scores and returns the best top-K trajectories.
    """

    def __init__(
        self,
        input_dim: int = 8,
        future_steps: int = 30,
        subgraph_hidden_dim: int = 64,
        subgraph_layers: int = 3,
        graph_dim: int = 128,
        global_graph_layers: int = 1,
        decoder_hidden_dim: int = 128,
        endpoint_hidden_dim: int = 128,
        tnt_top_m: int = 50,
        tnt_output_k: int = 1,
        score_hidden_dim: int = 128,
        dropout: float = 0.1,
        predict_offsets: bool = False,
        architecture: str = "paper",
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.tnt_top_m = int(tnt_top_m)
        self.tnt_output_k = int(tnt_output_k)
        self.predict_offsets = predict_offsets
        self.architecture = architecture
        self.use_paper_architecture = architecture == "paper"
        if self.tnt_top_m < 1:
            raise ValueError("tnt_top_m must be >= 1")
        if self.tnt_output_k < 1:
            raise ValueError("tnt_output_k must be >= 1")
        self.traj_subgraph = PolylineSubgraphEncoder(
            input_dim=input_dim,
            hidden_dim=subgraph_hidden_dim,
            num_layers=subgraph_layers,
        )
        if self.use_paper_architecture:
            self.map_subgraph = PolylineSubgraphEncoder(
                input_dim=input_dim,
                hidden_dim=subgraph_hidden_dim,
                num_layers=subgraph_layers,
            )
        else:
            self.map_subgraph = self.traj_subgraph
        self.global_graph = StackedGlobalAttentionGraph(
            self.traj_subgraph.output_dim,
            graph_dim=graph_dim,
            num_layers=global_graph_layers,
            use_layer_norm=True,
        )
        target_input_dim = graph_dim + 3
        self.target_head = nn.Sequential(
            nn.Linear(target_input_dim, endpoint_hidden_dim),
            nn.LayerNorm(endpoint_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(endpoint_hidden_dim, endpoint_hidden_dim),
            nn.ReLU(),
        )
        self.target_logit = nn.Linear(endpoint_hidden_dim, 1)
        self.target_offset = nn.Linear(endpoint_hidden_dim, 2)
        self.motion_decoder = nn.Sequential(
            nn.Linear(graph_dim + 2, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, future_steps * 2),
        )
        self.score_head = nn.Sequential(
            nn.Linear(graph_dim + future_steps * 2, score_hidden_dim),
            nn.LayerNorm(score_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden_dim, score_hidden_dim),
            nn.ReLU(),
            nn.Linear(score_hidden_dim, 1),
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

    def _decode_motion(self, target_feature: torch.Tensor, endpoints: torch.Tensor) -> torch.Tensor:
        if endpoints.ndim == 2:
            decoder_input = torch.cat([target_feature, endpoints], dim=-1)
            pred = self.motion_decoder(decoder_input).view(target_feature.shape[0], self.future_steps, 2)
            if self.predict_offsets:
                pred = torch.cumsum(pred, dim=1)
            return pred
        if endpoints.ndim != 3:
            raise ValueError(f"Expected endpoints [B,2] or [B,M,2], got {tuple(endpoints.shape)}")
        batch_size, num_targets, _ = endpoints.shape
        target_context = target_feature.unsqueeze(1).expand(-1, num_targets, -1)
        decoder_input = torch.cat([target_context, endpoints], dim=-1)
        pred = self.motion_decoder(decoder_input).view(batch_size, num_targets, self.future_steps, 2)
        if self.predict_offsets:
            pred = torch.cumsum(pred, dim=2)
        return pred

    def _score_trajectories(self, target_feature: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        batch_size, num_targets, _, _ = trajectories.shape
        target_context = target_feature.unsqueeze(1).expand(-1, num_targets, -1)
        score_input = torch.cat([target_context, trajectories.reshape(batch_size, num_targets, -1)], dim=-1)
        return self.score_head(score_input).squeeze(-1)

    def forward(
        self,
        polylines: torch.Tensor,
        polyline_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        polyline_types: torch.Tensor | None = None,
        target_candidates: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        target_endpoint: torch.Tensor | None = None,
        target_teacher_forcing: bool = False,
        output_k: int | None = None,
        return_target_details: bool = False,
    ):
        if polyline_types is None:
            raise ValueError("CandidateTNTVectorNetTrajectoryPredictor requires polyline_types.")
        if target_candidates is None or candidate_mask is None:
            raise ValueError("CandidateTNTVectorNetTrajectoryPredictor requires target_candidates and candidate_mask.")
        graph_features = self._encode(polylines, polyline_mask, segment_mask, polyline_types)
        target_feature = graph_features[:, 0]

        candidate_norm = torch.linalg.norm(target_candidates, dim=-1, keepdim=True)
        target_context = target_feature.unsqueeze(1).expand(-1, target_candidates.shape[1], -1)
        candidate_features = torch.cat([target_context, target_candidates, candidate_norm], dim=-1)
        hidden = self.target_head(candidate_features)
        target_logits = self.target_logit(hidden).squeeze(-1)
        target_logits = target_logits.masked_fill(~candidate_mask.bool(), torch.finfo(target_logits.dtype).min)
        target_offsets = self.target_offset(hidden)
        refined_targets = target_candidates + target_offsets

        top_m = min(self.tnt_top_m, target_candidates.shape[1])
        _, top_indices = target_logits.topk(top_m, dim=1)
        batch_idx = torch.arange(polylines.shape[0], device=polylines.device)
        selected_endpoints = refined_targets[batch_idx.unsqueeze(1), top_indices]
        selected_candidate_mask = candidate_mask.bool()[batch_idx.unsqueeze(1), top_indices]
        candidate_trajectories = self._decode_motion(target_feature, selected_endpoints)
        score_logits = self._score_trajectories(target_feature, candidate_trajectories)
        score_logits = score_logits.masked_fill(~selected_candidate_mask, torch.finfo(score_logits.dtype).min)

        top_score_index = score_logits.argmax(dim=1)
        predicted_endpoint = selected_endpoints[batch_idx, top_score_index]
        if target_teacher_forcing:
            if target_endpoint is None:
                raise ValueError("target_endpoint is required when target_teacher_forcing=True.")
            trajectory_with_gt = self._decode_motion(target_feature, target_endpoint)
        else:
            trajectory_with_gt = None

        selected_k = min(int(output_k or self.tnt_output_k), candidate_trajectories.shape[1])
        _, score_order = score_logits.topk(selected_k, dim=1)
        pred = candidate_trajectories[batch_idx.unsqueeze(1), score_order]
        if selected_k == 1:
            pred = pred[:, 0]

        if return_target_details:
            return pred, {
                "target_logits": target_logits,
                "target_offsets": target_offsets,
                "refined_targets": refined_targets,
                "selected_target_indices": top_indices,
                "selected_endpoints": selected_endpoints,
                "selected_candidate_mask": selected_candidate_mask,
                "candidate_trajectories": candidate_trajectories,
                "trajectory_score_logits": score_logits,
                "trajectory_with_gt": trajectory_with_gt,
                "predicted_endpoint": predicted_endpoint,
                "top_index": top_indices[batch_idx, top_score_index],
                "score_order": score_order,
            }
        return pred


class LSTMVectorNetTrajectoryPredictor(nn.Module):
    """Shared LSTM agent encoder plus VectorNet map/global graph.

    Agent polylines (type 0 target, type 1 neighbors) are encoded by one
    shared LSTM over their vector segments. Map polylines (type >= 2) keep the
    VectorNet local subgraph encoder. The resulting agent and map node features
    are fused by the same global attention graph used by VectorNet.
    """

    def __init__(
        self,
        input_dim: int = 8,
        future_steps: int = 30,
        agent_lstm_hidden_dim: int = 64,
        agent_lstm_layers: int = 1,
        subgraph_hidden_dim: int = 64,
        subgraph_layers: int = 3,
        graph_dim: int = 128,
        global_graph_layers: int = 1,
        decoder_hidden_dim: int = 128,
        decoder_type: str = "mlp",
        dropout: float = 0.1,
        predict_offsets: bool = True,
        num_modes: int = 1,
        auxiliary_node_loss: bool = False,
        aux_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.predict_offsets = predict_offsets
        self.auxiliary_node_loss = auxiliary_node_loss
        self.num_modes = int(num_modes)
        if self.num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        self.decoder_type = str(decoder_type).lower()
        if self.decoder_type not in {"mlp", "lstm"}:
            raise ValueError(f"decoder_type={decoder_type!r} is not supported; expected 'mlp' or 'lstm'.")
        self.agent_lstm = nn.LSTM(
            input_dim,
            agent_lstm_hidden_dim,
            num_layers=agent_lstm_layers,
            batch_first=True,
            dropout=dropout if agent_lstm_layers > 1 else 0.0,
        )
        self.map_subgraph = PolylineSubgraphEncoder(
            input_dim=input_dim,
            hidden_dim=subgraph_hidden_dim,
            num_layers=subgraph_layers,
        )
        polyline_feature_dim = self.map_subgraph.output_dim
        self.agent_projection = nn.Sequential(
            nn.Linear(agent_lstm_hidden_dim, polyline_feature_dim),
            nn.LayerNorm(polyline_feature_dim),
            nn.ReLU(),
        )
        self.global_graph = StackedGlobalAttentionGraph(
            polyline_feature_dim,
            graph_dim=graph_dim,
            num_layers=global_graph_layers,
            use_layer_norm=True,
        )
        self.mode_embedding = nn.Parameter(torch.zeros(self.num_modes, graph_dim)) if self.num_modes > 1 else None
        self.mode_score_head = nn.Linear(graph_dim, 1) if self.num_modes > 1 else None

        if self.decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(graph_dim, decoder_hidden_dim),
                nn.LayerNorm(decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(decoder_hidden_dim, future_steps * 2),
            )
            self.recurrent_decoder = None
            self.recurrent_output = None
        else:
            self.decoder = None
            self.recurrent_decoder = nn.LSTM(
                input_size=graph_dim,
                hidden_size=decoder_hidden_dim,
                batch_first=True,
            )
            self.recurrent_output = nn.Linear(decoder_hidden_dim, 2)
        if auxiliary_node_loss:
            self.aux_head = nn.Sequential(
                nn.Linear(graph_dim + 3, aux_hidden_dim),
                nn.LayerNorm(aux_hidden_dim),
                nn.ReLU(),
                nn.Linear(aux_hidden_dim, 4),
            )
        else:
            self.aux_head = None

    def _encode_agents(
        self,
        polylines: torch.Tensor,
        segment_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_polylines, max_segments, _feature_dim = polylines.shape
        flat_agent_mask = agent_mask.reshape(-1)
        encoded = polylines.new_zeros(
            batch_size * num_polylines,
            self.agent_lstm.hidden_size,
        )
        if not flat_agent_mask.any():
            return self.agent_projection(encoded).view(batch_size, num_polylines, -1)

        flat_polylines = polylines.reshape(batch_size * num_polylines, max_segments, -1)
        lengths = segment_mask.long().sum(dim=2).reshape(-1)
        agent_indices = flat_agent_mask.nonzero(as_tuple=False).squeeze(-1)
        agent_sequences = flat_polylines[agent_indices]
        agent_lengths = lengths[agent_indices].clamp_min(1).cpu()
        packed = pack_padded_sequence(
            agent_sequences,
            agent_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _out, (hidden, _cell) = self.agent_lstm(packed)
        encoded[agent_indices] = hidden[-1]
        return self.agent_projection(encoded).view(batch_size, num_polylines, -1)

    def _encode(
        self,
        polylines: torch.Tensor,
        polyline_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        polyline_types: torch.Tensor,
    ) -> torch.Tensor:
        segment_mask = segment_mask.bool()
        polyline_mask = polyline_mask.bool()
        agent_mask = polyline_mask & (polyline_types < 2)
        map_mask = polyline_mask & (polyline_types >= 2)

        agent_features = self._encode_agents(polylines, segment_mask, agent_mask)
        map_features = self.map_subgraph(polylines, segment_mask)
        polyline_features = torch.where(map_mask.unsqueeze(-1), map_features, agent_features)
        polyline_features = polyline_features * polyline_mask.unsqueeze(-1)
        return self.global_graph(polyline_features, polyline_mask)

    def _apply_auxiliary_mask(
        self,
        polylines: torch.Tensor,
        segment_mask: torch.Tensor,
        aux_mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = segment_mask.bool()
        random_mask = torch.rand(valid.shape, device=valid.device) < aux_mask_ratio
        aux_mask = valid & random_mask
        if not aux_mask.any():
            valid_indices = valid.nonzero(as_tuple=False)
            if valid_indices.numel() == 0:
                return polylines, aux_mask
            chosen = valid_indices[torch.randint(valid_indices.shape[0], (1,), device=valid.device)]
            aux_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
        masked_polylines = polylines.clone()
        masked_polylines[aux_mask, :6] = 0.0
        masked_polylines[aux_mask, 7] = 0.0
        return masked_polylines, aux_mask

    def _auxiliary_completion(
        self,
        graph_features: torch.Tensor,
        polylines: torch.Tensor,
        polyline_types: torch.Tensor,
        aux_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.aux_head is None:
            raise RuntimeError("auxiliary_node_loss is disabled for this model.")
        batch_idx, poly_idx, seg_idx = aux_mask.nonzero(as_tuple=True)
        context = graph_features[batch_idx, poly_idx]
        segment_pos = seg_idx.float().unsqueeze(-1) / max(polylines.shape[2] - 1, 1)
        type_feature = polyline_types[batch_idx, poly_idx].float().unsqueeze(-1) / 10.0
        valid_feature = torch.ones_like(segment_pos)
        aux_input = torch.cat([context, segment_pos, type_feature, valid_feature], dim=-1)
        pred = self.aux_head(aux_input)
        target = polylines[batch_idx, poly_idx, seg_idx, :4]
        return pred, target

    def forward(
        self,
        polylines: torch.Tensor,
        polyline_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        polyline_types: torch.Tensor | None = None,
        return_aux: bool = False,
        aux_mask_ratio: float = 0.15,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if polyline_types is None:
            raise ValueError("LSTMVectorNetTrajectoryPredictor requires polyline_types.")
        aux_data: dict[str, torch.Tensor] = {}
        model_input = polylines
        aux_mask = None
        if return_aux and self.auxiliary_node_loss:
            model_input, aux_mask = self._apply_auxiliary_mask(polylines, segment_mask, aux_mask_ratio)
        graph_features = self._encode(model_input, polyline_mask, segment_mask, polyline_types)
        target_feature = graph_features[:, 0]
        mode_scores = None
        if self.mode_embedding is not None:
            target_feature = target_feature.unsqueeze(1) + self.mode_embedding.unsqueeze(0)
            mode_scores = self.mode_score_head(target_feature).squeeze(-1)
        if self.decoder_type == "mlp":
            if target_feature.ndim == 2:
                pred = self.decoder(target_feature).view(polylines.shape[0], self.future_steps, 2)
            else:
                pred = self.decoder(target_feature).view(polylines.shape[0], self.num_modes, self.future_steps, 2)
        else:
            if target_feature.ndim == 2:
                decoder_in = target_feature.unsqueeze(1).expand(-1, self.future_steps, -1)
                decoded, _ = self.recurrent_decoder(decoder_in)
                pred = self.recurrent_output(decoded)
            else:
                batch_size, num_modes, graph_dim = target_feature.shape
                decoder_in = target_feature.reshape(batch_size * num_modes, graph_dim)
                decoder_in = decoder_in.unsqueeze(1).expand(-1, self.future_steps, -1)
                decoded, _ = self.recurrent_decoder(decoder_in)
                pred = self.recurrent_output(decoded).view(batch_size, num_modes, self.future_steps, 2)
        if self.predict_offsets:
            pred = torch.cumsum(pred, dim=-2)
        if self.num_modes == 1 and pred.ndim == 4:
            pred = pred[:, 0]
        if return_aux and self.auxiliary_node_loss and aux_mask is not None:
            aux_pred, aux_target = self._auxiliary_completion(graph_features, polylines, polyline_types, aux_mask)
            aux_data = {"pred": aux_pred, "target": aux_target}
            if mode_scores is not None:
                aux_data["mode_scores"] = mode_scores
            return pred, aux_data
        if mode_scores is not None:
            return pred, mode_scores
        return pred


class MiniTNTLSTMVectorNetTrajectoryPredictor(LSTMVectorNetTrajectoryPredictor):
    """LSTM + VectorNet encoder with a TNT-style endpoint-conditioned decoder.

    This is a lightweight single-target variant of TNT: the model predicts one
    endpoint from the target context, then conditions the trajectory decoder on
    that endpoint. It does not sample target candidates, produce top-M
    trajectories, or score trajectories.
    """

    def __init__(
        self,
        input_dim: int = 8,
        future_steps: int = 30,
        agent_lstm_hidden_dim: int = 64,
        agent_lstm_layers: int = 1,
        subgraph_hidden_dim: int = 64,
        subgraph_layers: int = 3,
        graph_dim: int = 128,
        global_graph_layers: int = 1,
        decoder_hidden_dim: int = 128,
        endpoint_hidden_dim: int = 128,
        dropout: float = 0.1,
        predict_offsets: bool = True,
        auxiliary_node_loss: bool = False,
        aux_hidden_dim: int = 128,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            future_steps=future_steps,
            agent_lstm_hidden_dim=agent_lstm_hidden_dim,
            agent_lstm_layers=agent_lstm_layers,
            subgraph_hidden_dim=subgraph_hidden_dim,
            subgraph_layers=subgraph_layers,
            graph_dim=graph_dim,
            global_graph_layers=global_graph_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
            predict_offsets=predict_offsets,
            auxiliary_node_loss=auxiliary_node_loss,
            aux_hidden_dim=aux_hidden_dim,
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(graph_dim, endpoint_hidden_dim),
            nn.LayerNorm(endpoint_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(endpoint_hidden_dim, endpoint_hidden_dim),
            nn.ReLU(),
            nn.Linear(endpoint_hidden_dim, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(graph_dim + 2, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, future_steps * 2),
        )

    def forward(
        self,
        polylines: torch.Tensor,
        polyline_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        polyline_types: torch.Tensor | None = None,
        return_aux: bool = False,
        aux_mask_ratio: float = 0.15,
        return_endpoint: bool = False,
        target_endpoint: torch.Tensor | None = None,
        endpoint_teacher_forcing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]:
        if polyline_types is None:
            raise ValueError("MiniTNTLSTMVectorNetTrajectoryPredictor requires polyline_types.")
        aux_data: dict[str, torch.Tensor] = {}
        model_input = polylines
        aux_mask = None
        if return_aux and self.auxiliary_node_loss:
            model_input, aux_mask = self._apply_auxiliary_mask(polylines, segment_mask, aux_mask_ratio)
        graph_features = self._encode(model_input, polyline_mask, segment_mask, polyline_types)
        target_feature = graph_features[:, 0]
        endpoint = self.endpoint_head(target_feature)
        decoder_endpoint = endpoint
        if endpoint_teacher_forcing:
            if target_endpoint is None:
                raise ValueError("target_endpoint is required when endpoint_teacher_forcing=True.")
            decoder_endpoint = target_endpoint
        decoder_input = torch.cat([target_feature, decoder_endpoint], dim=-1)
        pred = self.decoder(decoder_input).view(polylines.shape[0], self.future_steps, 2)
        if self.predict_offsets:
            pred = torch.cumsum(pred, dim=1)

        if return_aux and self.auxiliary_node_loss and aux_mask is not None:
            aux_pred, aux_target = self._auxiliary_completion(graph_features, polylines, polyline_types, aux_mask)
            aux_data = {"pred": aux_pred, "target": aux_target}
            if return_endpoint:
                return pred, aux_data, endpoint
            return pred, aux_data
        if return_endpoint:
            return pred, endpoint
        return pred


def build_vectornet_model(
    input_dim: int = 8,
    future_steps: int = 30,
    model_type: str = "vectornet",
    agent_lstm_hidden_dim: int = 64,
    agent_lstm_layers: int = 1,
    subgraph_hidden_dim: int = 64,
    subgraph_layers: int = 3,
    graph_dim: int = 128,
    global_graph_layers: int = 1,
    decoder_hidden_dim: int = 128,
    decoder_type: str = "mlp",
    dropout: float = 0.1,
    predict_offsets: bool = True,
    architecture: str = "current",
    num_modes: int = 1,
    auxiliary_node_loss: bool = False,
    aux_hidden_dim: int = 128,
    endpoint_hidden_dim: int = 128,
) -> nn.Module:
    if model_type == "lstm_vectornet":
        return LSTMVectorNetTrajectoryPredictor(
            input_dim=input_dim,
            future_steps=future_steps,
            agent_lstm_hidden_dim=agent_lstm_hidden_dim,
            agent_lstm_layers=agent_lstm_layers,
            subgraph_hidden_dim=subgraph_hidden_dim,
            subgraph_layers=subgraph_layers,
            graph_dim=graph_dim,
            global_graph_layers=global_graph_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_type=decoder_type,
            dropout=dropout,
            predict_offsets=predict_offsets,
            num_modes=num_modes,
            auxiliary_node_loss=auxiliary_node_loss,
            aux_hidden_dim=aux_hidden_dim,
        )
    if model_type == "mini_tnt_lstm_vectornet":
        return MiniTNTLSTMVectorNetTrajectoryPredictor(
            input_dim=input_dim,
            future_steps=future_steps,
            agent_lstm_hidden_dim=agent_lstm_hidden_dim,
            agent_lstm_layers=agent_lstm_layers,
            subgraph_hidden_dim=subgraph_hidden_dim,
            subgraph_layers=subgraph_layers,
            graph_dim=graph_dim,
            global_graph_layers=global_graph_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            endpoint_hidden_dim=endpoint_hidden_dim,
            dropout=dropout,
            predict_offsets=predict_offsets,
            auxiliary_node_loss=auxiliary_node_loss,
            aux_hidden_dim=aux_hidden_dim,
        )
    if model_type != "vectornet":
        raise ValueError(
            f"Unknown model_type={model_type!r}; "
            "expected 'vectornet', 'lstm_vectornet', or 'mini_tnt_lstm_vectornet'"
        )
    return VectorNetTrajectoryPredictor(
        input_dim=input_dim,
        future_steps=future_steps,
        subgraph_hidden_dim=subgraph_hidden_dim,
        subgraph_layers=subgraph_layers,
        graph_dim=graph_dim,
        global_graph_layers=global_graph_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        dropout=dropout,
        predict_offsets=predict_offsets,
        architecture=architecture,
        auxiliary_node_loss=auxiliary_node_loss,
        aux_hidden_dim=aux_hidden_dim,
    )
