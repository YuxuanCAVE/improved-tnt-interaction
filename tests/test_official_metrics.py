import torch

from improved_tnt.engine.official_metrics import official_minade_minfde_mr


def _constant_trajectory(final_x: float = 0.0, final_y: float = 0.0) -> torch.Tensor:
    trajectory = torch.zeros(1, 1, 30, 2)
    trajectory[0, 0, -1] = torch.tensor([final_x, final_y])
    return trajectory


def test_official_mr_uses_strict_lateral_threshold() -> None:
    target = torch.zeros(1, 30, 2)
    yaw = torch.zeros(1)
    speed = torch.zeros(1)

    _, _, at_threshold = official_minade_minfde_mr(
        _constant_trajectory(final_y=1.0), target, yaw, speed
    )
    _, _, above_threshold = official_minade_minfde_mr(
        _constant_trajectory(final_y=1.0001), target, yaw, speed
    )

    assert torch.equal(at_threshold, torch.tensor([0.0]))
    assert torch.equal(above_threshold, torch.tensor([1.0]))


def test_official_mr_uses_speed_dependent_longitudinal_threshold() -> None:
    target = torch.zeros(5, 30, 2)
    prediction = torch.zeros(5, 1, 30, 2)
    yaw = torch.zeros(5)
    speed = torch.tensor([0.0, 1.4, 6.2, 11.0, 15.0])
    thresholds = torch.tensor([1.0, 1.0, 1.5, 2.0, 2.0])
    prediction[:, 0, -1, 0] = thresholds

    _, _, at_threshold = official_minade_minfde_mr(prediction, target, yaw, speed)
    prediction[:, 0, -1, 0] += 0.0001
    _, _, above_threshold = official_minade_minfde_mr(prediction, target, yaw, speed)

    assert torch.equal(at_threshold, torch.zeros(5))
    assert torch.equal(above_threshold, torch.ones(5))


def test_official_mr_requires_all_modalities_to_miss() -> None:
    target = torch.zeros(2, 30, 2)
    prediction = torch.zeros(2, 2, 30, 2)
    prediction[:, 0, -1, 1] = 1.1
    prediction[0, 1, -1, 1] = 0.5
    prediction[1, 1, -1, 1] = 1.1

    _, _, miss = official_minade_minfde_mr(
        prediction,
        target,
        final_yaw=torch.zeros(2),
        final_speed=torch.zeros(2),
    )

    assert torch.equal(miss, torch.tensor([0.0, 1.0]))


def test_official_metrics_reject_non_finite_metric_state() -> None:
    prediction = torch.zeros(1, 1, 30, 2)
    target = torch.zeros(1, 30, 2)

    try:
        official_minade_minfde_mr(
            prediction,
            target,
            final_yaw=torch.tensor([float("nan")]),
            final_speed=torch.zeros(1),
        )
    except ValueError as error:
        assert "final_yaw contains NaN or Inf" in str(error)
    else:
        raise AssertionError("Expected non-finite metric state to raise ValueError")
