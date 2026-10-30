import pytest

from mistral_fork.mistral.data.pretokenized.pretokenized_utils import (
    ConstantWeightSchedule,
    LinearInterpolationWeightSchedule,
)


class TestTotalTokens:
    def test_constant_schedule(self) -> None:
        schedule = ConstantWeightSchedule(weight=0.5)
        assert schedule.compute_total_tokens(10000) == 5000.0

    def test_linear_schedule_horizon_before_mix_start(self) -> None:
        # Entirely in phase 1
        schedule = LinearInterpolationWeightSchedule(
            weight_init=0.8, weight_final=0.2, mix_start_tok=5000, mix_end_tok=8000
        )
        assert schedule.compute_total_tokens(3000) == 0.8 * 3000

    def test_linear_schedule_horizon_during_transition(self) -> None:
        # Horizon at midpoint: weight goes 0.8 -> 0.5, trapezoidal area
        schedule = LinearInterpolationWeightSchedule(
            weight_init=0.8, weight_final=0.2, mix_start_tok=0, mix_end_tok=10000
        )
        result = schedule.compute_total_tokens(5000)
        expected = ((0.8 + 0.5) / 2) * 5000
        assert result == pytest.approx(expected)

    def test_linear_schedule_full_horizon(self) -> None:
        # phase1: 0.8 * 2000 = 1600
        # transition: (0.8 + 0.2) / 2 * 6000 = 3000
        # phase2: 0.2 * 2000 = 400
        schedule = LinearInterpolationWeightSchedule(
            weight_init=0.8, weight_final=0.2, mix_start_tok=2000, mix_end_tok=8000
        )
        result = schedule.compute_total_tokens(10000)
        assert result == pytest.approx(1600 + 3000 + 400)

    def test_linear_schedule_equals_constant_when_weights_equal(self) -> None:
        const = ConstantWeightSchedule(weight=0.5)
        linear = LinearInterpolationWeightSchedule(
            weight_init=0.5, weight_final=0.5, mix_start_tok=1000, mix_end_tok=5000
        )
        assert linear.compute_total_tokens(10000) == pytest.approx(
            const.compute_total_tokens(10000)
        )

    def test_linear_schedule_step_function(self) -> None:
        # mix_start_tok == mix_end_tok
        schedule = LinearInterpolationWeightSchedule(
            weight_init=0.8, weight_final=0.2, mix_start_tok=5000, mix_end_tok=5000
        )
        # Before phase change: all weight_init
        assert schedule.compute_total_tokens(3000) == pytest.approx(0.8 * 3000)
        # After phase change: phase1 + phase2 at weight_final
        assert schedule.compute_total_tokens(10000) == pytest.approx(
            0.8 * 5000 + 0.2 * 5000
        )

    def test_linear_schedule_horizon_at_mix_end(self) -> None:
        # Full interpolation: (1.0 + 0.0) / 2 * 10000 = 5000
        schedule = LinearInterpolationWeightSchedule(
            weight_init=1.0, weight_final=0.0, mix_start_tok=0, mix_end_tok=10000
        )
        assert schedule.compute_total_tokens(10000) == pytest.approx(5000.0)


if __name__ == "__main__":
    pytest.main([__file__])
