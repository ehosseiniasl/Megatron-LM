# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Arg handling in ``apply_determinism_to_args``.

Built through the real parser rather than a hand-made Namespace. That is the
point: every guard here rejects a config, so each one has to name an option the
operator can actually set, and the flag it names has to exist. A Namespace built
by hand agrees with whatever the guard believes and proves neither.

``moe_router_aux_loss_fusion`` is the case that motivated this. Its CLI flag is
generated from the dataclass field rather than written by hand, so the spelling
follows the field's default and moves if that default changes.

No GPU and no process group: parsing and validation both run on CPU.
"""

import argparse

import pytest

from megatron.training.arguments import add_megatron_arguments
from megatron.training.determinism import apply_determinism_to_args


def _args(*flags):
    """Parse ``flags`` with the real Megatron parser, under --deterministic-mode."""
    parser = argparse.ArgumentParser()
    add_megatron_arguments(parser)
    args, _ = parser.parse_known_args(["--deterministic-mode", *flags])
    assert args.deterministic_mode, "--deterministic-mode did not survive parsing"
    return args


class TestMoeRouterAuxLossFusion:
    """The aux loss reaches the backward via MoEAuxLossAutoScaler; fused TopK routing
    only selects experts. So the two fuse independently and only the former is refused.
    """

    def test_fused_routing_alone_is_refused_because_the_aux_loss_follows_it(self):
        with pytest.raises(AssertionError, match="moe_router_aux_loss_fusion"):
            apply_determinism_to_args(_args("--moe-router-fusion"))

    def test_opting_the_aux_loss_out_keeps_fused_routing_available(self):
        args = _args("--moe-router-fusion", "--no-moe-router-aux-loss-fusion")
        apply_determinism_to_args(args)
        assert args.moe_router_fusion is True
        assert args.moe_router_aux_loss_fusion is False

    def test_neither_flag_is_accepted(self):
        args = _args()
        apply_determinism_to_args(args)
        assert args.moe_router_aux_loss_fusion is None  # unset: follows moe_router_fusion


class TestRejectionsNameASettableFlag:
    """A guard that names an option with no CLI flag leaves the operator stuck: the
    run is refused and the message asks for something they cannot set.
    """

    def test_every_rejected_arg_has_a_flag_that_turns_it_off(self):
        parser = argparse.ArgumentParser()
        add_megatron_arguments(parser)
        by_dest = {a.dest: a.option_strings for a in parser._actions}

        for dest in ("cross_entropy_loss_fusion", "tp_comm_overlap", "moe_router_aux_loss_fusion"):
            assert by_dest.get(
                dest
            ), f"{dest} is rejected under --deterministic-mode but has no flag"

    def test_the_aux_loss_opt_out_flag_is_spelled_as_the_test_configs_expect(self):
        """Pins the generated spelling. It is derived from the field's default
        (``Optional[bool] = None``), so changing that default silently renames the
        flag and breaks every model_config.yaml that passes it.
        """
        args = _args("--no-moe-router-aux-loss-fusion")
        assert args.moe_router_aux_loss_fusion is False
