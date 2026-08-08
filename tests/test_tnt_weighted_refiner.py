from __future__ import annotations

import torch

from improved_tnt.models.tnt_refinement import TNTWeightedTrajectoryRefiner


def test_refiner_initialises_as_identity() -> None:
    model = TNTWeightedTrajectoryRefiner(graph_dim=8, future_steps=3, dropout=0.0)
    context = torch.randn(2, 8)
    base = torch.randn(2, 3, 2)

    refined, residual = model(context, base)

    assert torch.equal(refined, base)
    assert torch.count_nonzero(residual) == 0


def test_refiner_residual_is_bounded() -> None:
    model = TNTWeightedTrajectoryRefiner(
        graph_dim=8,
        future_steps=3,
        dropout=0.0,
        residual_limit=0.02,
    )
    with torch.no_grad():
        model.decoder[-1].bias.fill_(100.0)
    refined, residual = model(torch.randn(1, 8), torch.zeros(1, 3, 2))

    assert residual.abs().max() <= 0.02
    assert torch.allclose(refined, residual)
