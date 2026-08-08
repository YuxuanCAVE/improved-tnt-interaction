from __future__ import annotations

from argparse import Namespace
from typing import Any

from torch import nn

from improved_tnt.models.tnt import build_tnt_model


POLYLINE_MODEL_TYPES = {"tnt_vectornet"}


def model_family(model_type: str) -> str:
    if model_type in POLYLINE_MODEL_TYPES:
        return "polyline"
    expected = sorted(POLYLINE_MODEL_TYPES)
    raise ValueError(f"Unknown model_type={model_type!r}; expected one of {expected}")


def checkpoint_model_type(checkpoint: dict[str, Any]) -> str:
    ckpt_args = checkpoint.get("args", {})
    return str(ckpt_args.get("model_type", checkpoint.get("model_type", "lstm")))


def build_model_for_dataset(args: Namespace, dataset: Any) -> nn.Module:
    model_type = str(getattr(args, "model_type", "lstm"))
    model_family(model_type)
    if model_type == "tnt_vectornet":
        return build_tnt_model(
            input_dim=dataset.feature_dim,
            future_steps=dataset.future_steps,
            subgraph_hidden_dim=int(getattr(args, "subgraph_hidden_dim", 64)),
            subgraph_layers=int(getattr(args, "subgraph_layers", 3)),
            graph_dim=int(getattr(args, "graph_dim", 128)),
            global_graph_layers=int(getattr(args, "global_graph_layers", 2)),
            target_hidden_dim=int(getattr(args, "target_hidden_dim", getattr(args, "endpoint_hidden_dim", 128))),
            motion_hidden_dim=int(getattr(args, "motion_hidden_dim", getattr(args, "decoder_hidden_dim", 128))),
            score_hidden_dim=int(getattr(args, "score_hidden_dim", getattr(args, "decoder_hidden_dim", 128))),
            tnt_top_m=int(getattr(args, "tnt_top_m", 50)),
            tnt_output_k=int(getattr(args, "tnt_output_k", 1)),
            target_offset_limit=float(getattr(args, "target_offset_limit", 0.05)),
            dropout=float(getattr(args, "dropout", 0.1)),
            predict_offsets=bool(getattr(args, "predict_offsets", False)),
            architecture=str(getattr(args, "architecture", "paper")),
            use_refined_targets=bool(getattr(args, "use_refined_targets", True)),
            endpoint_exact_residual=bool(getattr(args, "endpoint_exact_residual", False)),
            trajectory_nms_threshold=float(getattr(args, "trajectory_nms_threshold_m", 2.0))
            / max(float(getattr(dataset, "coordinate_scale_m", getattr(args, "sensor_range_m", 1.0))), 1e-6),
        )
    raise ValueError(f"Unsupported TNT repository model_type={model_type!r}")
