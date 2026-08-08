from __future__ import annotations

import torch

from improved_tnt.models.tnt import TNTMotionEstimation


def test_endpoint_exact_motion_reaches_each_conditioning_endpoint() -> None:
    torch.manual_seed(7)
    model = TNTMotionEstimation(
        graph_dim=8,
        future_steps=6,
        hidden_dim=16,
        dropout=0.0,
        predict_offsets=True,
        endpoint_exact_residual=True,
    )
    target_feature = torch.randn(3, 8)
    endpoints = torch.randn(3, 4, 2)

    trajectories = model(target_feature, endpoints)

    assert trajectories.shape == (3, 4, 6, 2)
    torch.testing.assert_close(trajectories[:, :, -1], endpoints, atol=1e-6, rtol=1e-6)


def test_endpoint_exact_motion_supports_single_endpoint_and_old_weights() -> None:
    legacy = TNTMotionEstimation(
        graph_dim=8,
        future_steps=6,
        hidden_dim=16,
        dropout=0.0,
        predict_offsets=True,
        endpoint_exact_residual=False,
    )
    endpoint_exact = TNTMotionEstimation(
        graph_dim=8,
        future_steps=6,
        hidden_dim=16,
        dropout=0.0,
        predict_offsets=True,
        endpoint_exact_residual=True,
    )
    endpoint_exact.load_state_dict(legacy.state_dict(), strict=True)
    target_feature = torch.randn(3, 8)
    endpoints = torch.randn(3, 2)

    trajectories = endpoint_exact(target_feature, endpoints)

    assert trajectories.shape == (3, 6, 2)
    torch.testing.assert_close(trajectories[:, -1], endpoints, atol=1e-6, rtol=1e-6)
