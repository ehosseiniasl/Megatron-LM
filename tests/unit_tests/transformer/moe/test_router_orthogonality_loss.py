# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    router_orthogonality_loss_func,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

NUM_EXPERTS, HIDDEN = 8, 16
OFF_DIAGONAL = NUM_EXPERTS**2 - NUM_EXPERTS


def _reference(weight: torch.Tensor) -> torch.Tensor:
    """R^T R - I, built the slow way: normalize the rows of R^T, then take the Gram matrix."""
    weight = weight.double()
    weight = weight / weight.norm(dim=1, keepdim=True)
    gram = weight @ weight.t()
    return gram - torch.diag(gram.diagonal())


def _random_weight(dtype=torch.float32, seed=0):
    """A gate matrix with deliberately non-unit row norms, so row normalization matters."""
    torch.manual_seed(seed)
    return (torch.randn(NUM_EXPERTS, HIDDEN, dtype=dtype) * 0.1) * torch.linspace(
        0.5, 4.0, NUM_EXPERTS, dtype=dtype
    ).unsqueeze(1)


class TestRouterOrthogonalityLossFunc:
    def test_defaults_to_squared_frobenius(self):
        weight = _random_weight()
        assert torch.equal(
            router_orthogonality_loss_func(weight, 1.0),
            router_orthogonality_loss_func(weight, 1.0, norm="fro_sq"),
        )

    @pytest.mark.parametrize("norm", ["l1", "fro", "fro_sq"])
    def test_matches_reference(self, norm):
        weight = _random_weight()
        off_diagonal = _reference(weight)
        expected = {
            "l1": off_diagonal.abs().sum(),
            "fro": off_diagonal.square().sum().sqrt(),
            "fro_sq": off_diagonal.square().sum(),
        }[norm]

        loss = router_orthogonality_loss_func(weight, 0.5, norm=norm)
        assert torch.allclose(loss.double(), 0.5 * expected, rtol=1e-5, atol=1e-6)

    def test_diagonal_contributes_nothing(self):
        """Row normalization makes the Gram diagonal exactly one, so I cancels it exactly."""
        weight = _random_weight()
        gram = _reference(weight)
        assert torch.allclose(gram.diagonal(), torch.zeros(NUM_EXPERTS, dtype=torch.float64))
        orthogonal = torch.eye(NUM_EXPERTS, HIDDEN) * 3.0
        assert router_orthogonality_loss_func(orthogonal, 1.0, norm="l1").abs().item() < 1e-6

    @pytest.mark.parametrize("norm", ["l1", "fro", "fro_sq"])
    def test_invariant_to_row_scaling(self, norm):
        """The defining property of the mandatory row normalization."""
        weight = _random_weight()
        torch.manual_seed(1)
        scales = torch.rand(NUM_EXPERTS, 1) * 10 + 0.1
        base = router_orthogonality_loss_func(weight, 1.0, norm=norm)
        scaled = router_orthogonality_loss_func(weight * scales, 1.0, norm=norm)
        assert torch.allclose(base, scaled, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("norm", ["l1", "fro", "fro_sq"])
    def test_gradient_is_orthogonal_to_each_row(self, norm):
        """Row normalization projects out the radial component, so norms are left alone."""
        weight = torch.nn.Parameter(_random_weight(dtype=torch.float64))
        router_orthogonality_loss_func(weight, 1.0, norm=norm).backward()
        radial = (weight.grad * weight.detach()).sum(dim=1)
        assert torch.allclose(radial, torch.zeros(NUM_EXPERTS, dtype=torch.float64), atol=1e-12)
        assert weight.grad.abs().sum() > 0, "gradient vanished entirely"

    @pytest.mark.parametrize("norm", ["l1", "fro", "fro_sq"])
    def test_gradient_matches_autograd_through_the_reference(self, norm):
        weight = torch.nn.Parameter(_random_weight(dtype=torch.float64))
        router_orthogonality_loss_func(weight, 0.25, norm=norm).backward()

        reference_weight = torch.nn.Parameter(weight.detach().clone())
        off_diagonal = _reference(reference_weight)
        {
            "l1": off_diagonal.abs().sum(),
            "fro": off_diagonal.square().sum().sqrt(),
            "fro_sq": off_diagonal.square().sum(),
        }[norm].mul(0.25).backward()
        assert torch.allclose(weight.grad, reference_weight.grad, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize(
        "norm,divisor", [("l1", OFF_DIAGONAL), ("fro", OFF_DIAGONAL**0.5), ("fro_sq", OFF_DIAGONAL)]
    )
    def test_mean_divides_by_off_diagonal_count(self, norm, divisor):
        weight = _random_weight()
        total = router_orthogonality_loss_func(weight, 1.0, norm=norm, mean=False)
        averaged = router_orthogonality_loss_func(weight, 1.0, norm=norm, mean=True)
        assert torch.allclose(averaged, total / divisor, rtol=1e-5, atol=1e-7)

    def test_mean_is_the_mean_squared_cosine(self):
        weight = _random_weight()
        normalized = weight.double() / weight.double().norm(dim=1, keepdim=True)
        cosines = (normalized @ normalized.t())[~torch.eye(NUM_EXPERTS, dtype=torch.bool)]
        averaged = router_orthogonality_loss_func(weight, 1.0, norm="fro_sq", mean=True)
        assert torch.allclose(averaged.double(), cosines.square().mean(), rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("norm", ["l1", "fro", "fro_sq"])
    def test_zero_for_orthogonal_rows(self, norm):
        """The Q of a full-rank Gaussian matrix is orthonormal, so the off-diagonal is zero."""
        torch.manual_seed(0)
        weight = torch.linalg.qr(torch.randn(HIDDEN, NUM_EXPERTS, dtype=torch.float64))[0]
        loss = router_orthogonality_loss_func(weight.t().contiguous(), 1.0, norm=norm)
        assert loss.abs().item() < 1e-10

    def test_gradient_pushes_towards_orthogonality(self):
        weight = torch.nn.Parameter(_random_weight(dtype=torch.float64))
        before = router_orthogonality_loss_func(weight, 1.0, norm="fro_sq")
        before.backward()
        with torch.no_grad():
            stepped = weight - 1e-3 * weight.grad
        after = router_orthogonality_loss_func(stepped, 1.0, norm="fro_sq")
        assert after.item() < before.item()

    @pytest.mark.parametrize("norm", ["l1", "fro", "fro_sq"])
    @pytest.mark.parametrize("scale", [0.0, 1e-30, 1e-20])
    def test_degenerate_gate_row_stays_finite(self, norm, scale):
        """A gate row at or below the row-norm floor must stay finite and must not dominate."""
        weight = _random_weight()
        weight[3] *= scale
        weight = torch.nn.Parameter(weight)
        loss = router_orthogonality_loss_func(weight, 1.0, norm=norm)
        loss.backward()
        assert torch.isfinite(loss), (norm, scale, loss)
        assert torch.isfinite(weight.grad).all(), (norm, scale)
        healthy = router_orthogonality_loss_func(weight.detach()[:3], 1.0, norm=norm)
        assert loss.item() < 10 * max(healthy.item(), 1e-6)

    @pytest.mark.parametrize("norm", ["l1", "fro", "fro_sq"])
    def test_floor_leaves_small_but_representable_rows_alone(self, norm):
        """The floor must not attenuate a direction float32 still represents.

        A gate row norm of 1e-5 is orders below anything a trained router reaches, yet its
        squared norm of 1e-10 is still well clear of the 1e-12 floor, so scale invariance holds.
        """
        weight = _random_weight()
        direction = weight[3] / weight[3].norm()
        unit, small = weight.clone(), weight.clone()
        unit[3], small[3] = direction, direction * 1e-5
        assert torch.allclose(
            router_orthogonality_loss_func(unit, 1.0, norm=norm),
            router_orthogonality_loss_func(small, 1.0, norm=norm),
            rtol=1e-4,
            atol=1e-7,
        )

    def test_rejects_unknown_norm(self):
        with pytest.raises(ValueError, match="Unsupported router orthogonality loss norm"):
            router_orthogonality_loss_func(_random_weight(), 1.0, norm="nuclear")

    @pytest.mark.parametrize("norm", ["l1", "fro", "fro_sq"])
    def test_does_not_retain_a_hidden_size_copy(self, norm):
        """Row normalization must stay in E x E space.

        An upcast or row-normalized [num_moe_experts, hidden_size] copy saved for backward would
        dominate the activation memory this term adds, once per MoE layer.
        """
        weight = torch.nn.Parameter(_random_weight(dtype=torch.bfloat16))
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (saved.append(t), t)[1], lambda t: t
        ):
            router_orthogonality_loss_func(weight, 1.0, norm=norm)
        copies = [t for t in saved if t.shape == weight.shape and t.data_ptr() != weight.data_ptr()]
        assert not copies, f"{norm} saved a [num_experts, hidden_size] copy for backward"


@pytest.fixture
def router_factory(monkeypatch):
    """Use the real router and autograd attachment without distributed collectives."""
    tracker = MoEMetricsTracker()
    monkeypatch.setattr(
        "megatron.core.transformer.moe.router.get_moe_metrics_tracker", lambda: tracker
    )
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    group = SimpleNamespace(size=lambda: 1)
    groups = SimpleNamespace(tp=group, cp=group, tp_cp=group, tp_dp_cp=group)

    def build(limit, *, is_mtp_layer=False, **kwargs):
        config = TransformerConfig(
            num_layers=6,
            hidden_size=HIDDEN,
            num_attention_heads=2,
            num_moe_experts=NUM_EXPERTS,
            use_cpu_initialization=True,
            add_bias_linear=False,
            moe_router_orthogonality_loss_coeff=0.5,
            moe_router_orthogonality_loss_num_layers=limit,
            **kwargs,
        )
        router = TopKRouter(config, pg_collection=groups, is_mtp_layer=is_mtp_layer)
        with torch.no_grad():
            router.weight.copy_(_random_weight())
        return router, tracker

    return build


@pytest.mark.parametrize(
    "frequency,layer_number,selected",
    [
        (1, 2, True),
        (1, 3, False),
        (2, 3, True),
        (2, 5, False),
        ([0, 1, 0, 1, 1, 0], 4, True),
        ([0, 1, 0, 1, 1, 0], 5, False),
    ],
)
def test_limit_counts_moe_layers_and_gates_aux_gradient(
    router_factory, frequency, layer_number, selected
):
    router, tracker = router_factory(2, moe_layer_freq=frequency)
    router.set_layer_number(layer_number)
    logits = torch.zeros(4, NUM_EXPERTS, requires_grad=True)
    output = router.apply_router_orthogonality_loss(logits)
    (output.sum() * 0).backward()

    assert router.layer_number == layer_number  # Logging keeps the global block number.
    if selected:
        reference = router.weight.detach().clone().requires_grad_(True)
        router_orthogonality_loss_func(reference, 0.5).backward()
        torch.testing.assert_close(router.weight.grad, reference.grad)
        assert router.weight.grad.abs().sum() > 0
        values = tracker.metrics["router_orthogonality_loss"].values
        assert values.nonzero().flatten().tolist() == [layer_number - 1]
    else:
        assert output is logits
        assert router.weight.grad is None
        assert not tracker.metrics


@pytest.mark.parametrize("is_mtp_layer", [False, True])
@pytest.mark.parametrize("limit", [None, 0, 100])
def test_default_zero_and_oversized_limit(router_factory, is_mtp_layer, limit):
    router, tracker = router_factory(
        limit, is_mtp_layer=is_mtp_layer, mtp_num_layers=2, mtp_use_repeated_layer=True
    )
    router.set_layer_number(1 if is_mtp_layer else 6)
    logits = torch.zeros(4, NUM_EXPERTS, requires_grad=True)
    output = router.apply_router_orthogonality_loss(logits)
    (output.sum() * 0).backward()

    selected = limit is None or (limit > 0 and not is_mtp_layer)
    if selected:
        reference = router.weight.detach().clone().requires_grad_(True)
        router_orthogonality_loss_func(reference, 0.25 if is_mtp_layer else 0.5).backward()
        torch.testing.assert_close(router.weight.grad, reference.grad)
        assert tracker.metrics
    else:
        assert output is logits
        assert router.weight.grad is None
        assert not tracker.metrics


def test_limit_requires_layer_number(router_factory):
    router, _ = router_factory(2)
    with pytest.raises(ValueError, match="global MoE layer number"):
        router.apply_router_orthogonality_loss(torch.zeros(4, NUM_EXPERTS))
