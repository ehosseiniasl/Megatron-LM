"""Teach the legacy Megatron TikTokenizer to read Mistral Tekken files."""

import json
import os
import socket
import sys
from pathlib import Path


def _inject_model_args(saved_args: dict[str, object]) -> None:
    """Inject model-shape args before Megatron computes derived configuration."""
    model_keys = {
        "add-bias-linear",
        "bf16",
        "disable-bias-linear",
        "ffn-hidden-size",
        "group-query-attention",
        "hidden-size",
        "hidden-dropout",
        "hybrid-layer-pattern",
        "init-method-std",
        "kv-channels",
        "mamba-head-dim",
        "mamba-num-groups",
        "mamba-num-heads",
        "mamba-state-dim",
        "max-position-embeddings",
        "moe-layer-freq",
        "moe-ffn-hidden-size",
        "moe-grouped-gemm",
        "moe-permute-fusion",
        "moe-router-enable-expert-bias",
        "moe-router-score-function",
        "moe-router-topk",
        "moe-router-topk-scaling-factor",
        "moe-shared-expert-intermediate-size",
        "normalization",
        "norm-epsilon",
        "no-rope-fusion",
        "no-use-layer-wise-param-layout",
        "num-attention-heads",
        "num-experts",
        "num-layers",
        "num-query-groups",
        "padded-vocab-size",
        "position-embedding-type",
        "rotary-base",
        "seq-length",
        "squared-relu",
        "untie-embeddings-and-output-weights",
        "use-fused-weighted-squared-relu",
        "vocab-size",
    }
    model_keys.update(key for key in saved_args if key.startswith("wide-residual-"))

    injected = []
    for key in model_keys:
        value = saved_args.get(key)
        if value is None or value is False:
            continue
        option = f"--{key}"
        if option in sys.argv:
            continue
        injected.append(option)
        if not isinstance(value, bool):
            injected.append(str(value))
    sys.argv.extend(injected)


def _ensure_single_rank_process_group() -> None:
    import torch.distributed as dist

    if dist.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(port_socket.getsockname()[1])
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


def install_mistral_tekken_compat(megatron_args_json: str) -> None:
    """Install support for the wrapped Tekken JSON format used by Thunderdome."""
    from megatron.core.tokenizers.text.libraries import tiktoken_tokenizer

    tokenizer_class = tiktoken_tokenizer.TikTokenTokenizer
    if getattr(tokenizer_class, "_mistral_tekken_compat", False):
        return

    original_init = tokenizer_class.__init__
    original_reload = tiktoken_tokenizer.reload_mergeable_ranks

    def reload_mergeable_ranks(path, max_vocab=None, num_special_tokens=None):
        with open(path, encoding="utf-8") as tokenizer_file:
            payload = json.load(tokenizer_file)
        if not isinstance(payload, dict) or not isinstance(payload.get("vocab"), list):
            return original_reload(path, max_vocab, num_special_tokens)

        vocab = payload["vocab"]
        if max_vocab is not None:
            vocab = vocab[:max_vocab]

        import base64

        ranks = {}
        for index, token in enumerate(vocab):
            assert token.keys() == {"rank", "token_bytes", "token_str"}
            assert token["rank"] == index
            token_bytes = base64.b64decode(token["token_bytes"])
            assert index >= 256 or token_bytes == bytes([index])
            ranks[token_bytes] = index + num_special_tokens
        assert len(ranks) == len(vocab)
        return ranks

    def compatible_init(self, tokenizer_path, **kwargs):
        with open(tokenizer_path, encoding="utf-8") as tokenizer_file:
            payload = json.load(tokenizer_file)
        if not isinstance(payload, dict) or not isinstance(payload.get("vocab"), list):
            original_init(self, tokenizer_path, **kwargs)
            return

        config = payload["config"]
        special_tokens = [token["token_str"] for token in payload["special_tokens"]]
        assert [token["rank"] for token in payload["special_tokens"]] == list(
            range(len(special_tokens))
        )
        assert len(special_tokens) == config["default_num_special_tokens"]

        # The legacy constructor requires three obsolete special-token names.
        # Supply temporary aliases for validation, then restore the exact Tekken
        # special-token table and rebuild the tiktoken Encoding below.
        constructor_tokens = special_tokens.copy()
        for offset, required_token in enumerate(("<mask>", "<cls>", "<sep>"), start=1):
            if required_token not in constructor_tokens:
                constructor_tokens[-offset] = required_token

        kwargs.update(
            pattern="v2",
            vocab_size=config["default_vocab_size"],
            num_special_tokens=config["default_num_special_tokens"],
            special_tokens=constructor_tokens,
        )
        original_init(self, tokenizer_path, **kwargs)

        special_tokens_dict = {token: index for index, token in enumerate(special_tokens)}
        self.special_tokens = special_tokens
        self.special_filler = []
        self._unk_id = special_tokens_dict["<unk>"]
        self._bos_id = special_tokens_dict["<s>"]
        self._eos_id = special_tokens_dict["</s>"]
        self._pad_id = special_tokens_dict["<pad>"]
        self._mask_id = special_tokens_dict.get("<mask>", self._unk_id)
        self._cls_id = special_tokens_dict.get("<cls>", self._unk_id)
        self._sep_id = special_tokens_dict.get("<sep>", self._unk_id)
        self.shifted_id2token = {
            **{index: token for index, token in enumerate(special_tokens)},
            **self.id2token,
        }
        self.tokenizer = tiktoken_tokenizer.tiktoken.Encoding(
            name=Path(tokenizer_path).parent.name,
            pat_str=config["pattern"],
            mergeable_ranks=self.token2id,
            special_tokens=special_tokens_dict,
        )
        self._vocab = special_tokens_dict | self.token2id

    tiktoken_tokenizer.reload_mergeable_ranks = reload_mergeable_ranks
    tokenizer_class.__init__ = compatible_init
    tokenizer_class._mistral_tekken_compat = True

    from megatron.training import arguments

    with open(megatron_args_json, encoding="utf-8") as args_file:
        saved_args = json.load(args_file)
    original_parse = arguments.parse_and_validate_args

    def compatible_parse_and_validate_args(*args, **kwargs):
        _inject_model_args(saved_args)
        parsed_args = original_parse(*args, **kwargs)
        _ensure_single_rank_process_group()
        return parsed_args

    arguments.parse_and_validate_args = compatible_parse_and_validate_args
