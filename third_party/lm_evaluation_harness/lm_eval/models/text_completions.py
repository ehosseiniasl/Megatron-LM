import hashlib
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import lm_eval
import lm_eval.utils as utils
import numpy as np
from lm_eval.api.instance import LLTokenRequest
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import OpenaiCompletionsLM
from lm_eval.utils import eval_logger
from tqdm import tqdm


@register_model("notok-completions")
class TokenizerFreeLocalCompletionsLM(OpenaiCompletionsLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        tokenizer_backend: Literal["api"] = "api",
        truncate: bool = False,
        max_gen_toks: int = 1024,
        batch_size: int = 1,
        seed: int = -1,
        max_length: Optional[int] = 4096,
        tokenizer: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize MegatronCompletionsLM with tokenizer-free approach.

        :param base_url: str
            Base URL for the Megatron API endpoint
        :param tokenizer_backend: str
            Must be "api" for Megatron
        :param max_length: int
            Rolling token window size.
        :param tokenizer: str
            Optional Hugging Face tokenizer ID or local path. Required for true
            token-stride rolling windows.
        """
        # Initialize with parent class but force tokenizer_backend to "api"
        super().__init__(
            model=model,
            base_url=base_url,
            tokenizer=tokenizer,
            tokenizer_backend=tokenizer_backend,
            truncate=truncate,
            max_gen_toks=max_gen_toks,
            batch_size=batch_size,
            seed=seed,
            max_length=max_length,
            **kwargs,
        )
        self.token_cache = {}
        self._hf_tokenizer = None

        if tokenizer:
            try:
                from huggingface_hub import snapshot_download
                from transformers import AutoTokenizer

                tokenizer_path = tokenizer
                if not Path(tokenizer).is_dir():
                    tokenizer_path = snapshot_download(tokenizer, local_files_only=True)
                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                eval_logger.info(
                    f"notok-completions: loaded tokenizer {tokenizer} for token-stride rolling windows"
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to load tokenizer={tokenizer!r} for notok-completions. "
                    "A valid local Hugging Face tokenizer is required for rolling token windows."
                ) from exc

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Encode string using MD5 hash as token IDs."""
        md5_tok = int(hashlib.md5(string.encode()).hexdigest(), 16)
        tokens = [md5_tok] * max(1, len(string))

        if md5_tok not in self.token_cache:
            self.token_cache[md5_tok] = string
        else:
            assert self.token_cache[md5_tok] == string
        return tokens

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode tokens back to string using token cache."""
        assert tokens[-1] in self.token_cache, "Token not found in cache"
        return self.token_cache[tokens[-1]]

    def _loglikelihood_tokens(
        self, requests: list[LLTokenRequest], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        tokens_list = []
        all_chunks = []

        for chunk in lm_eval.models.utils.chunks(
            re_ord.get_reordered(), self.batch_size
        ):
            chunk_tok_inps, chunk_text_inps = [], []
            chunk_tokens_list = []
            for (context, continuation), context_toks, continuation_toks in chunk:
                tok_inp = (context_toks + continuation_toks)[-(self.max_length + 1) :]
                chunk_tok_inps.append(tok_inp)
                chunk_text_inps.append(context + continuation)
                chunk_tokens_list.append(tok_inp)

            all_chunks.append(
                (
                    (
                        chunk_text_inps
                        if self.tokenizer_backend == "api"
                        else chunk_tok_inps
                    ),
                    {},
                )
            )
            tokens_list.extend(chunk_tokens_list)

        responses = self.load_balancer.submit_requests(
            requests=all_chunks,
            total_requests=len(requests),
            disable_tqdm=disable_tqdm,
            cache_hook=self.cache_hook,
            model=self.model,
            echo=True,
            max_tokens=0,
            temperature=0.0,
            logprobs=1,
            seed=self.seed,
            **({"extra_body": {"add_BOS": True}} if self.add_bos else {}),
        )

        # Flatten responses
        flattened_responses = []
        for response in responses:
            flattened_responses.extend(response.choices)

        # Parse responses
        for i, choice in enumerate(flattened_responses):
            token = tokens_list[i]
            context = self.token_cache[token[0]]
            complete = self.token_cache[token[-1]]
            assert token[0] != token[-1], "Tokens are not unique"

            tok_offsets = choice.logprobs.text_offset
            idx = np.searchsorted(tok_offsets, len(context))
            ctxlen, contlen = idx, len(tok_offsets) - idx

            assert ctxlen > 0, "Context length must be greater than 0"
            assert contlen > 0, "Completion length must be greater than 0"

            logprobs = sum(choice.logprobs.token_logprobs[-contlen:])

            # Check if generation was greedy
            is_greedy = True
            for i in range(ctxlen, len(choice.logprobs.token_logprobs)):
                token_text = choice.logprobs.tokens[i]
                top_tokens = choice.logprobs.top_logprobs[i] or {}
                top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x], default=None)
                if top_token != token_text:
                    is_greedy = False
                    break

            res.append((logprobs, is_greedy))

        return re_ord.get_original(res)

    def parse_logprobs(
        self,
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
    ) -> List[Tuple[float, bool]]:
        """Parse logprobs from API response."""
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            for tok, choice in zip(tokens, out.choices):
                context = self.token_cache[tok[0]]
                complete = self.token_cache[tok[-1]]
                assert tok[0] != tok[-1], "Tokens are not unique"

                tok_offsets = choice.logprobs.text_offset
                idx = np.searchsorted(tok_offsets, len(context))
                ctxlen, contlen = idx, len(tok_offsets) - idx

                assert ctxlen > 0, "Context length must be greater than 0"
                assert contlen > 0, "Completion length must be greater than 0"

                logprobs = sum(choice.logprobs.token_logprobs[-contlen:])

                # Check if generation was greedy
                is_greedy = True
                for i in range(ctxlen, len(choice.logprobs.token_logprobs)):
                    token = choice.logprobs.tokens[i]
                    top_tokens = choice.logprobs.top_logprobs[i]
                    top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
                    if top_token != token:
                        is_greedy = False
                        break

                res.append((logprobs, is_greedy))
        return res

    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> List[float]:
        if self._hf_tokenizer is None:
            raise ValueError(
                "notok-completions loglikelihood_rolling requires tokenizer=<hf id or local path> "
                "in --model_args so rolling windows can be measured in real tokens."
            )

        return self._rolling_token_stride(requests, disable_tqdm)

    def _rolling_token_stride(
        self, requests, disable_tqdm: bool = False
    ) -> List[float]:
        tok = self._hf_tokenizer

        all_windows: List[str] = []
        all_context_chars: List[int] = []
        window_counts: List[int] = []

        for req in tqdm(requests, disable=disable_tqdm):
            (string,) = req.args
            assert isinstance(string, str)
            window_tokens = self._rolling_token_window(req) - 2
            stride_tokens = window_tokens // 2
            ids = tok.encode(string, add_special_tokens=False)
            doc_start = len(all_windows)
            pos = 0
            n = len(ids)

            while True:
                ctx_start = max(0, pos - stride_tokens)
                new_end = min(pos + stride_tokens, n)
                window_text = tok.decode(
                    ids[ctx_start:new_end],
                    skip_special_tokens=True,
                )
                if not window_text:
                    if new_end >= n:
                        break
                    pos += stride_tokens
                    continue

                context_text = (
                    tok.decode(ids[ctx_start:pos], skip_special_tokens=True)
                    if pos > ctx_start
                    else ""
                )
                all_windows.append(window_text)
                all_context_chars.append(len(context_text))

                if new_end >= n:
                    break
                pos += stride_tokens

            window_counts.append(len(all_windows) - doc_start)

        return self._submit_and_score(
            all_windows,
            all_context_chars,
            window_counts,
            disable_tqdm,
        )

    def _submit_and_score(
        self,
        all_windows: List[str],
        all_context_chars: List[int],
        window_counts: List[int],
        disable_tqdm: bool,
    ) -> List[float]:
        if not all_windows:
            return [0.0] * len(window_counts)

        batch_requests = []
        for i in range(0, len(all_windows), self.batch_size):
            batch = all_windows[i : i + self.batch_size]
            batch_requests.append((batch, {}))

        responses = self.load_balancer.submit_requests(
            requests=batch_requests,
            total_requests=len(all_windows),
            disable_tqdm=disable_tqdm,
            cache_hook=self.cache_hook,
            model=self.model,
            echo=True,
            max_tokens=0,
            temperature=0.0,
            logprobs=1,
            seed=self.seed,
            **({"extra_body": {"add_BOS": True}} if self.add_bos else {}),
        )

        all_nlls = []
        ctx_iter = iter(all_context_chars)
        for response_batch in responses:
            for choice in response_batch.choices:
                context_chars = next(ctx_iter)
                tokens = choice.logprobs.tokens or []
                logprobs = choice.logprobs.token_logprobs or []
                # Without BOS, the first visible token has no log-probability.
                # With BOS, the server returns its conditioned log-probability.
                start_idx = 0 if self.add_bos else 1
                if context_chars > 0:
                    cum = 0
                    for idx in range(start_idx, len(tokens)):
                        cum += len(tokens[idx])
                        if cum > context_chars:
                            start_idx = idx
                            break
                    else:
                        start_idx = len(tokens)
                all_nlls.append(
                    sum(logprob for logprob in logprobs[start_idx:] if logprob is not None)
                )

        if len(all_nlls) != len(all_windows):
            eval_logger.warning(
                f"Expected {len(all_windows)} results but got {len(all_nlls)}. "
                "Some windows may have failed."
            )
            all_nlls.extend([0.0] * (len(all_windows) - len(all_nlls)))

        loglikelihoods = []
        start_idx = 0
        for count in window_counts:
            if count == 0:
                loglikelihoods.append(0.0)
                continue
            loglikelihoods.append(sum(all_nlls[start_idx : start_idx + count]))
            start_idx += count

        return loglikelihoods

    def _rolling_token_window(self, request) -> int:
        token_window = getattr(request, "token_window", None)
        if token_window is None:
            token_window = self.max_length

        try:
            token_window = int(token_window)
        except (TypeError, ValueError) as exc:
            raise ValueError("rolling token_window must be an integer") from exc

        if token_window < 4:
            raise ValueError("rolling token_window must be at least 4")

        return token_window
