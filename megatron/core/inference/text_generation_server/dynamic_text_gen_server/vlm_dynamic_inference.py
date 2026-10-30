# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for VLM dynamic-batching inference.

Exposes the small surface that the dynamic text generation server needs to
support multimodal checkpoints:

  * :func:`add_vlm_inference_args` — argparse group for VLM-specific args
  * :func:`_detect_vlm_from_checkpoint` — peek at saved training args and
    decide GPT-vs-VLM, with CLI > checkpoint > parser-default precedence
  * :func:`_print_resolved_args` — diagnostic dump of the args namespace
    *after* the late checkpoint resolution above
  * :func:`resolve_vlm_special_tokens` — resolve the image token IDs from the
    pretokenized config, checkpoint args, or tokenizer, requiring agreement
  * :func:`get_model` — build and load either a GPT or LLaVA model

The image-preprocessing helpers live in :mod:`.image_preprocessing` and are
re-exported here for backwards compatibility with older standalone callers.
"""

import json
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.transformer.module import MegatronModule
from megatron.training import get_args
from megatron.training import get_model as _get_model
from megatron.training import print_rank_0
from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint

# NOTE: ``get_model`` below does a ``from model import model_provider`` for the
# ``examples/multimodal/model.py`` file, whose siblings use bare imports like
# ``from config import ...``. The *caller* of this module (typically
# ``tools/run_dynamic_text_generation_server.py``) is expected to have already
# added the repo root and ``examples/multimodal/`` to ``sys.path`` before
# invoking ``get_model``. ``megatron/core`` does not mutate ``sys.path`` here.


def add_vlm_inference_args(parser):
    """Add VLM-specific inference arguments on top of the standard inference args."""
    from megatron.inference.utils import add_inference_args

    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="VLM dynamic inference")
    group.add_argument(
        "--input-image-path",
        type=str,
        default=None,
        help="Path to input image(s). Can be a single image or directory.",
    )
    group.add_argument(
        "--input-prompts-json",
        type=str,
        default=None,
        help="Path to JSON file with prompts and image paths. "
        'Format: [{"prompt": "...", "image": "path/to/image.jpg"}, ...]',
    )
    group.add_argument(
        "--vision-projection-type",
        choices=("mlp", "mlp2", "affine"),
        default="mlp",
        help="Projection reconstructed for MIMO checkpoints.",
    )
    group.add_argument("--vision-projection-ffn-hidden-size", type=int, default=None)
    group.add_argument("--vision-dynamic-resolution-max-patches", type=int, default=None)
    group.add_argument("--load-vision-from", type=str, default=None)
    group.add_argument("--vision-encoder-checkpoint-root", type=str, default=None)
    group.add_argument("--use-packed-sequence", action="store_true", default=False)
    group.add_argument(
        "--inference-allow-gtp-padding",
        action="store_true",
        default=False,
        help=(
            "Allow model weights saved with GTP alignment padding to load into a non-GTP "
            "inference topology. Use only for trusted GTP checkpoints."
        ),
    )
    return parser


_MISSING = object()


def _jsonable_arg_value(value):
    if value is _MISSING:
        return "<MISSING>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_arg_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_arg_value(val) for key, val in value.items()}
    return repr(value)


def _arg_value_changed(before, after):
    return _jsonable_arg_value(before) != _jsonable_arg_value(after)


def _print_resolved_args(title, args):
    """Print args after late checkpoint resolution.

    Megatron's standard args table is emitted during parse/initialize, before
    this server copies VLM-only fields out of the checkpoint. Print a second
    table at the point where the values are the ones model construction will
    actually consume, followed by a per-attr provenance dump showing where
    each VLM-relevant value came from (CLI, checkpoint, parser default).
    """
    print_rank_0(f'------------------------ {title} ------------------------')
    str_list = []
    for arg in vars(args):
        if arg.startswith("_"):
            continue
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print_rank_0(arg)
    print_rank_0(f'-------------------- end of {title} ---------------------')

    resolution = getattr(args, "_vlm_arg_resolution", None)
    if not resolution:
        return

    print_rank_0("---------------- VLM argument provenance ----------------")
    for record in resolution:
        attr = record["attr"]
        final_value = getattr(args, attr, _MISSING)
        changed_after_resolution = _arg_value_changed(record["resolved_value"], final_value)
        payload = {
            "arg": attr,
            "source": record["source"],
            "parser": _jsonable_arg_value(record["parser_value"]),
            "checkpoint": _jsonable_arg_value(record["checkpoint_value"]),
            "resolved": _jsonable_arg_value(record["resolved_value"]),
            "final": _jsonable_arg_value(final_value),
            "parser_changed_by_resolution": record["parser_changed_by_resolution"],
            "checkpoint_overridden": record["checkpoint_overridden"],
            "changed_after_resolution": changed_after_resolution,
            "note": record["note"],
        }
        print_rank_0(f"[vlm_arg_provenance] {json.dumps(payload, sort_keys=True)}")
    print_rank_0("------------ end of VLM argument provenance -------------")


@dataclass
class VLMSpecialTokens:
    """Semantic VLM image token IDs resolved for inference.

    Only IDs are carried. Checkpoints trained on pre-tokenized Mistral data
    serve with a ``NullTokenizer``, so no token strings are available; the
    break and end IDs are ``None`` when no source supplies them.
    """

    image_token_id: int
    image_break_token_id: Optional[int] = None
    image_end_token_id: Optional[int] = None

    @property
    def supports_native_image_tokens(self) -> bool:
        """Native row-terminated layout needs both the break and end token IDs."""
        return self.image_break_token_id is not None and self.image_end_token_id is not None


def _get_huggingface_tokenizer(tokenizer):
    """Unwrap the Megatron tokenizer down to the underlying HF tokenizer."""
    tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
    return getattr(tokenizer, "tokenizer", tokenizer)


def _resolve_named_special_token(tokenizer, name: str) -> Optional[Tuple[str, int]]:
    """Resolve one named VLM special token and assert it round-trips to a single ID.

    Returns ``None`` when the tokenizer declares neither the token nor its ID.
    Tokenizers that declare only one of the pair are misconfigured and raise.
    """
    token = getattr(tokenizer, name, None)
    token_id = getattr(tokenizer, f"{name}_id", None)
    if token is None and token_id is None:
        return None
    if token is None or token_id is None:
        raise ValueError(
            f"VLM tokenizer must define both {name!r} and {name + '_id'!r} through "
            "Hugging Face extra_special_tokens"
        )

    token = str(token)
    if not isinstance(token_id, int):
        raise ValueError(f"VLM tokenizer {name + '_id'!r} must be an integer, got {token_id!r}")
    if token not in tokenizer.all_special_tokens or token_id not in tokenizer.all_special_ids:
        raise ValueError(f"VLM tokenizer {name!r}={token!r} must be registered as a special token")
    if tokenizer.convert_tokens_to_ids(token) != token_id:
        raise ValueError(f"VLM tokenizer {name!r}={token!r} does not map to token ID {token_id}")
    encoded_ids = tokenizer.encode(token, add_special_tokens=False)
    if encoded_ids != [token_id]:
        raise ValueError(
            f"VLM tokenizer {name!r}={token!r} must encode to [{token_id}], got "
            f"{encoded_ids!r}; ensure split_special_tokens=False"
        )
    return token, token_id


_IMAGE_TOKEN_FIELDS = ("image_token_id", "image_break_token_id", "image_end_token_id")


def _pretokenized_image_token_ids(args):
    """Read the image token IDs from the Mistral pretokenized config, if configured.

    This is the authoritative source for checkpoints trained on pre-tokenized
    Mistral data, which use a ``NullTokenizer`` and therefore have no tokenizer
    to carry the token identities.

    Deliberately reads the raw mapping rather than calling
    ``load_pretokenized_config``. That helper validates the full training-data
    schema and asserts the current ``CONFIG_VERSION``, so a config written by an
    older training run fails on fields inference never uses (a v8 config, for
    example, predates the required ``drop_bos``). Only the three image token IDs
    matter here, and they have been stable across schema versions.
    """
    config_path = getattr(args, "pretokenized_config", None)
    if not config_path:
        return {}

    from pathlib import Path

    from mistral_adapter.pretokenized_config_utils import _load_pretokenized_config_mapping

    raw = _load_pretokenized_config_mapping(Path(config_path))
    if raw.get("skip_images", False):
        return {}

    ids = {}
    for field, config_key in (
        ("image_token_id", "img_id"),
        ("image_break_token_id", "img_break_id"),
        ("image_end_token_id", "img_end_id"),
    ):
        value = raw.get(config_key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"Pretokenized config {config_path}: {config_key!r} must be an "
                f"integer token ID, got {value!r}"
            )
        ids[field] = value
    return ids


def _tokenizer_image_token_ids(tokenizer):
    """Read image token IDs from a Hugging Face tokenizer's ``extra_special_tokens``."""
    tokenizer = _get_huggingface_tokenizer(tokenizer)
    ids = {}
    for name in ("image_token", "image_break_token", "image_end_token"):
        resolved = _resolve_named_special_token(tokenizer, name)
        if resolved is not None:
            ids[f"{name}_id"] = resolved[1]
    return ids


def resolve_vlm_special_tokens(tokenizer, args=None) -> VLMSpecialTokens:
    """Resolve the image token IDs and cross-check every source that supplies them.

    Sources are consulted in order of authority:

    1. the Mistral pretokenized config named by ``args.pretokenized_config``;
    2. explicit ``args`` values (from the CLI or restored from the checkpoint);
    3. a Hugging Face tokenizer declaring them via ``extra_special_tokens``.

    Whichever sources are present must agree. A silent disagreement misplaces
    image embeddings and produces plausible but wrong output, so any conflict
    raises rather than picking a winner. A missing checkpoint-era pretokenized
    config is ignored only when the tokenizer supplies all three IDs itself.

    Args:
        tokenizer: Megatron or Hugging Face tokenizer used to serve the model.
        args: Resolved argument namespace. When ``None``, only the tokenizer is
            consulted.

    Returns:
        The resolved :class:`VLMSpecialTokens`. ``image_break_token_id`` and
        ``image_end_token_id`` are ``None`` when no source supplies them, in
        which case native image-token expansion must stay disabled.

    Raises:
        ValueError: If sources disagree, or if no source supplies the image
            token ID at all.
    """
    sources = {}
    tokenizer_ids = _tokenizer_image_token_ids(tokenizer)
    if args is not None:
        try:
            pretokenized = _pretokenized_image_token_ids(args)
        except FileNotFoundError:
            if not all(field in tokenizer_ids for field in _IMAGE_TOKEN_FIELDS):
                raise
            print_rank_0(
                "Checkpoint pretokenized config is unavailable; using the complete "
                "VLM special-token mapping from the Hugging Face tokenizer."
            )
            pretokenized = {}
        if pretokenized:
            sources["pretokenized config"] = pretokenized
        explicit = {
            field: getattr(args, field)
            for field in _IMAGE_TOKEN_FIELDS
            if getattr(args, field, None) is not None
        }
        if explicit:
            sources["checkpoint/CLI args"] = explicit

    if tokenizer_ids:
        sources["tokenizer"] = tokenizer_ids

    resolved = {}
    for field in _IMAGE_TOKEN_FIELDS:
        for source_name, values in sources.items():
            value = values.get(field)
            if value is None:
                continue
            if field in resolved and resolved[field] != value:
                raise ValueError(
                    f"Conflicting {field}: "
                    + ", ".join(
                        f"{name}={vals[field]!r}"
                        for name, vals in sources.items()
                        if vals.get(field) is not None
                    )
                    + ". Refusing to guess which source describes the checkpoint."
                )
            resolved.setdefault(field, value)

    if resolved.get("image_token_id") is None:
        raise ValueError(
            "Could not determine the VLM image token ID. Provide it through "
            "--pretokenized-config, an image_token_id recorded in the checkpoint, "
            "or a Hugging Face tokenizer declaring extra_special_tokens."
        )

    return VLMSpecialTokens(
        image_token_id=resolved["image_token_id"],
        image_break_token_id=resolved.get("image_break_token_id"),
        image_end_token_id=resolved.get("image_end_token_id"),
    )


def _clamp_sequence_parallel(args):
    """Re-apply ``validate_args``' sequence-parallel rule after late resolution.

    ``_detect_vlm_from_checkpoint`` runs *after* ``parse_and_validate_args``, so
    a ``sequence_parallel`` value recovered from the checkpoint bypasses the
    ``tensor_model_parallel_size == 1`` guard in
    :func:`megatron.training.arguments.validate_args`. Without this, serving a
    hybrid or Mamba checkpoint at TP=1 leaves ``sequence_parallel=True`` in the
    ``TransformerConfig``, which has no matching assertion and therefore fails
    silently rather than loudly.
    """
    if getattr(args, "sequence_parallel", False) and args.tensor_model_parallel_size == 1:
        print_rank_0(
            "Disabling sequence parallelism recovered from the checkpoint because "
            "tensor model parallelism is disabled"
        )
        args.sequence_parallel = False


def _detect_vlm_from_checkpoint(args, user_passed_attrs=None):
    """Peek at the checkpoint's saved training args to detect VLM vs GPT.

    Returns True if the checkpoint was trained as a LLaVA or MIMO VLM, False
    otherwise. As a side-effect, selects the text model provider or copies
    VLM-specific args from the checkpoint into the current args namespace.
    VLM resolution provenance is recorded on ``args._vlm_arg_resolution`` for
    the diagnostic dump.

    Precedence for each attr is CLI > checkpoint > parser default. Callers
    pass ``user_passed_attrs`` to indicate which attribute names the user
    actually typed on the command line; those values are left alone.
    """
    user_passed_attrs = user_passed_attrs or set()
    result = load_args_from_checkpoint(args)
    if not isinstance(result, tuple):
        return False

    _, checkpoint_args = result
    mimo_provider = getattr(checkpoint_args, 'model_provider', None)
    is_mimo = mimo_provider in ('nemotron-moe-vlm', 'nemotron-moe-mistral-vit')
    if getattr(checkpoint_args, 'language_model_type', None) is None and not is_mimo:
        if "model_provider" not in user_passed_attrs:
            if mimo_provider in ("gpt", "hybrid", "mamba"):
                args.model_provider = mimo_provider
            elif any(
                (
                    getattr(checkpoint_args, "is_hybrid_model", False),
                    getattr(checkpoint_args, "hybrid_layer_pattern", None),
                    getattr(checkpoint_args, "hybrid_override_pattern", None),
                    getattr(checkpoint_args, "mamba_num_heads", None),
                )
            ):
                args.model_provider = "hybrid"

        if "sequence_parallel" not in user_passed_attrs:
            checkpoint_sequence_parallel = getattr(checkpoint_args, "sequence_parallel", None)
            if checkpoint_sequence_parallel is not None:
                args.sequence_parallel = checkpoint_sequence_parallel
            elif getattr(args, "model_provider", None) in ("hybrid", "mamba"):
                args.sequence_parallel = True
            _clamp_sequence_parallel(args)
        return False

    if is_mimo:
        args._vlm_is_mimo = True
        vision_encoder_type = getattr(checkpoint_args, 'vision_encoder_type', None)
        if vision_encoder_type is None:
            vision_encoder_type = {
                'nemotron-moe-vlm': 'radio',
                'nemotron-moe-mistral-vit': 'mistral-vit',
            }[mimo_provider]
        if vision_encoder_type == 'radio':
            vision_model_type = (
                'post-c-radio-omni'
                if getattr(checkpoint_args, 'dynamic_resolution', False)
                else 'radio'
            )
            encoder_module = 'radio_encoder'
            encoder_path = 'encoders.radio_encoder.radio_model.'
        elif vision_encoder_type == 'mistral-vit':
            model_size = getattr(checkpoint_args, 'mistral_vit_model_size', '410m')
            vision_model_type = (
                'mistral-vit-1p6b' if model_size in ('1.6b', '1p6b') else 'mistral-vit'
            )
            encoder_module = 'mistral_vit_encoder'
            encoder_path = 'encoders.mistral_vit_encoder.'
        else:
            raise ValueError(
                f'Unsupported MIMO vision encoder type {vision_encoder_type!r} '
                f'for provider {mimo_provider!r}'
            )

        mimo_defaults = {
            'language_model_type': 'nemotron6-moe',
            'vision_model_type': vision_model_type,
            'disable_vision_class_token': True,
            'freeze_ViT': getattr(checkpoint_args, 'freeze_vit', False),
        }
        if vision_encoder_type == 'mistral-vit':
            mimo_defaults.update(
                {
                    'encoder_num_layers': 30 if vision_model_type == 'mistral-vit-1p6b' else 24,
                    'spatial_merge_size': 2,
                    'dynamic_resolution': True,
                    'dynamic_resolution_max_patches': 12100,
                }
            )
        for attr, value in mimo_defaults.items():
            if getattr(checkpoint_args, attr, None) is None:
                setattr(checkpoint_args, attr, value)

        modality_prefix = f'modality_submodules.{encoder_module}.module.module.'
        args._vlm_checkpoint_prefix_map = {
            'language_model.': 'language_model.module.module.',
            'vision_model.': f'{modality_prefix}{encoder_path}',
            'vision_projection.': f'{modality_prefix}input_projections.0.',
        }
        args._vlm_checkpoint_model_prefixes = ('language_model.module.module.', modality_prefix)
    else:
        args._vlm_is_mimo = False
        args._vlm_checkpoint_prefix_map = None
        args._vlm_checkpoint_model_prefixes = None

    vlm_attrs = [
        'language_model_type',
        'vision_model_type',
        'model_provider',
        'vision_encoder_type',
        'mistral_vit_model_size',
        'vision_projection_type',
        'vision_projection_ffn_hidden_size',
        'encoder_num_layers',
        'decoder_seq_length',
        'use_te',
        'disable_vision_class_token',
        'pixel_shuffle',
        'use_tile_tags',
        'max_num_tiles',
        'use_thumbnail',
        'use_tiling',
        'img_h',
        'img_w',
        'patch_dim',
        'spatial_merge_size',
        'tokenizer_prompt_format',
        'recompute_vision',
        'num_frames',
        'freeze_LM',
        'freeze_ViT',
        'allow_missing_vision_projection_checkpoint',
        'pixel_mean',
        'pixel_std',
        'use_area_weighted_aspect_ratio',
        'dynamic_resolution',
        'dynamic_resolution_min_patches',
        'dynamic_resolution_max_patches',
        'dynamic_resolution_min_side',
        'dynamic_resolution_max_side',
        'vision_dynamic_resolution_max_patches',
        'class_token_len',
        'radio_force_cpe_eval_mode',
        'radio_force_eval_mode',
        'radio_interpolate_only_cpe',
        'radio_cpe_aspect_ratio_select',
        'radio_disable_cpe',
        'spec',
        'transformer_impl',
        'is_hybrid_model',
        'hybrid_override_pattern',
        'hybrid_layer_pattern',
        'num_experts',
        'moe_ffn_hidden_size',
        'moe_router_topk',
        'moe_shared_expert_intermediate_size',
        'image_token_id',
        'image_break_token_id',
        'image_end_token_id',
        # The Mistral pretokenized config is the authoritative source for the
        # image / image-break / image-end IDs in this repository; the tokenizer
        # does not carry them as Hugging Face extra_special_tokens.
        'pretokenized_config',
    ]
    resolution = []
    for attr in vlm_attrs:
        parser_value = getattr(args, attr, _MISSING)
        checkpoint_value = getattr(checkpoint_args, attr, _MISSING)
        if attr in user_passed_attrs:
            source = "cli"
            resolved_value = getattr(args, attr, _MISSING)
            note = "explicit CLI value preserved"
        elif checkpoint_value is not _MISSING and checkpoint_value is not None:
            source = "checkpoint"
            setattr(args, attr, checkpoint_value)
            resolved_value = checkpoint_value
            note = "copied from checkpoint"
        elif checkpoint_value is None:
            source = "default"
            resolved_value = getattr(args, attr, _MISSING)
            note = "checkpoint value is None; kept parser/default value"
        else:
            source = "default"
            resolved_value = getattr(args, attr, _MISSING)
            note = "not present in checkpoint; kept parser/default value"

        resolution.append(
            {
                "attr": attr,
                "source": source,
                "parser_value": parser_value,
                "checkpoint_value": checkpoint_value,
                "resolved_value": resolved_value,
                "parser_changed_by_resolution": _arg_value_changed(parser_value, resolved_value),
                "checkpoint_overridden": (
                    source == "cli"
                    and checkpoint_value is not _MISSING
                    and checkpoint_value is not None
                    and _arg_value_changed(checkpoint_value, resolved_value)
                ),
                "note": note,
            }
        )

    args._vlm_arg_resolution = resolution

    return True


def _validate_checkpoint_model_mismatches(
    checkpoint_model_prefixes: tuple[str, ...], missing_keys: set[str], unexpected_keys: set[str]
) -> None:
    """Reject incomplete VLM model loads while tolerating TE extra-state differences."""

    def is_extra_state(key: str) -> bool:
        return key.rsplit('.', 1)[-1].split('/', 1)[0] == '_extra_state'

    unused_checkpoint_model_keys = {
        key
        for key in missing_keys
        if key.startswith(checkpoint_model_prefixes) and not is_extra_state(key)
    }
    absent_runtime_model_keys = {key for key in unexpected_keys if not is_extra_state(key)}
    if unused_checkpoint_model_keys or absent_runtime_model_keys:
        raise CheckpointingException(
            'VLM checkpoint model-parameter audit failed. '
            f'Checkpoint tensors not consumed by the model: {unused_checkpoint_model_keys}. '
            f'Runtime model tensors absent from the checkpoint: {absent_runtime_model_keys}.'
        )

    print_rank_0('VLM checkpoint model-parameter audit passed')


def _allow_gtp_padding_shape_mismatch(sharded_state_dict) -> None:
    """Permit DCP to crop trusted GTP padding while retaining key validation."""

    def mark_weight(value):
        if isinstance(value, ShardedTensor) and value.key.endswith('.weight'):
            value.allow_shape_mismatch = True
        return value

    dict_list_map_inplace(mark_weight, sharded_state_dict)


def get_model(is_vlm: bool) -> MegatronModule:
    """Build and load the model; dispatches to the right model_provider."""
    args = get_args()

    if is_vlm:
        if getattr(args, '_vlm_is_mimo', False):
            from pretrain_vlm import model_provider
        else:
            from model import model_provider  # examples/multimodal/model.py

        model = _get_model(partial(model_provider), wrap_with_ddp=False)
    else:
        from gpt_builders import gpt_builder  # examples/inference/gpt
        from model_provider import model_provider

        model = _get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)

    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    checkpoint_load_kwargs = {}
    checkpoint_model_prefixes = getattr(args, '_vlm_checkpoint_model_prefixes', None)
    if checkpoint_model_prefixes is not None:
        checkpoint_load_kwargs = {
            'model_sharded_state_dict_prefix_map': getattr(
                args, '_vlm_checkpoint_prefix_map', None
            ),
            'checkpoint_mismatch_validator': partial(
                _validate_checkpoint_model_mismatches, checkpoint_model_prefixes
            ),
        }
    if getattr(args, 'inference_allow_gtp_padding', False):
        checkpoint_load_kwargs['model_sharded_state_dict_modifier'] = (
            _allow_gtp_padding_shape_mismatch
        )
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=not args.inference_ckpt_non_strict,
        **checkpoint_load_kwargs,
    )

    assert len(model) == 1, "Virtual PP not supported for VLM inference"
    model = model[0]
    model.eval()
    return model
