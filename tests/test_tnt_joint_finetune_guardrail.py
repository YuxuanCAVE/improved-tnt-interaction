from scripts.joint_finetune_tnt_k1_guarded import is_guardrail_eligible


def test_k6_guardrail_accepts_metric_at_tolerance_boundary() -> None:
    assert is_guardrail_eligible(0.0368, 0.0348, 0.002)


def test_k6_guardrail_rejects_excessive_degradation() -> None:
    assert not is_guardrail_eligible(0.03681, 0.0348, 0.002)
