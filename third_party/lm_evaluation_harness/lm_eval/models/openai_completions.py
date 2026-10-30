import copy
import json
import os
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple

from tqdm import tqdm

import lm_eval.models.utils
from lm_eval import utils
from lm_eval.api.instance import Instance, LLTokenRequest
from lm_eval.api.model import LM, TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.load_balancer import LoadBalancedEndpoints
from lm_eval.utils import eval_logger


def get_result(response, ctxlen: int) -> Tuple[float, bool]:
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = response.logprobs.token_logprobs
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(response.logprobs.token_logprobs)):
        token = response.logprobs.tokens[i]
        top_tokens = response.logprobs.top_logprobs[i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        # print(top_token, token)
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


class TokenizerAPI:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _request(self, endpoint: str, data: dict) -> dict:
        import requests

        resp = requests.get(f"{self.base_url}/{endpoint}", json=data, timeout=30)
        if resp.status_code != 200:
            err_msg = f"{resp.status_code}: {resp.text}\n{data} failed at /{endpoint}\n"
            raise ValueError(err_msg)
        return resp.json()

    def encode(self, string: str) -> list[int]:
        return self._request("encode", {"text": string})["tokens"]

    def decode(self, tokens: list[int]) -> str:
        return self._request("decode", {"tokens": tokens})["text"]


@register_model("openai-completions", "local-completions")
class OpenaiCompletionsLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        model: str,
        base_url: str = None,
        tokenizer: Optional[str] = None,
        tokenizer_backend: Literal["tiktoken", "huggingface"] = "tiktoken",
        truncate: bool = False,
        max_gen_toks: int = 256,
        batch_size: int = 1,
        seed: int = -1,
        max_length: Optional[int] = None,
        max_concurrent_per_client: int = 1,
        add_bos: bool = False,
    ) -> None:
        """

        :param engine: str
            OpenAI API engine (e.g. gpt-3.5-turbo-instruct)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        assert seed == -1
        self.seed = seed
        try:
            import openai  # noqa: E401
            import tiktoken
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .\"[openai]\"`",
            )
        self.model = model
        self.base_url = base_url
        self.tokenizer_backend = tokenizer_backend
        self.truncate = truncate
        self._batch_size = int(batch_size)
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self.add_bos = (
            add_bos
            if isinstance(add_bos, bool)
            else str(add_bos).lower() in {"1", "true", "yes"}
        )
        self.load_balancer = LoadBalancedEndpoints(base_url, int(max_concurrent_per_client))

        # if we have a local model, use HF tokenizer over tiktoken
        if self.tokenizer_backend == "huggingface":
            import transformers  # noqa: E401

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else self.model
            )
            self.vocab_size = self.tokenizer.vocab
            self.end_of_text_token_id = self.tokenizer.eos_token
        elif self.tokenizer_backend == "tiktoken":
            if self.base_url:
                eval_logger.warning(
                    f"Passed `base_url={self.base_url}` but using Tiktoken tokenizer backend. "
                    "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                )

            self.tokenizer = tiktoken.encoding_for_model(self.model)
            self.vocab_size = self.tokenizer.n_vocab
            self.end_of_text_token_id = self.tokenizer.eot_token
        elif self.tokenizer_backend == "api":
            self.tokenizer = TokenizerAPI(base_url=self.base_url)
            self.end_of_text_token_id = None  # TODO
        else:
            raise ValueError(
                f"Expected tokenizer_backend to be one of ['tiktoken', 'huggingface'] but got {self.tokenizer_backend}"
            )

    @property
    def eot_token_id(self):
        return self.end_of_text_token_id

    @property
    def max_length(self) -> int:
        if self._max_length:
            return self._max_length
        else:
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(
        self, requests: list[LLTokenRequest], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        if not requests:
            return []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)
        all_chunks = []
        all_chunk_metadata = []

        for chunk in lm_eval.models.utils.chunks(
            re_ord.get_reordered(), self.batch_size
        ):
            inps = []
            ctxlens = []
            chunk_metadata = []
            for cache_key, context_toks, continuation_toks in chunk:
                # max_length+1 because the API takes up to 2049 tokens, including the first context token
                inp = (context_toks + continuation_toks)[-(self.max_length + 1) :]
                # TODO: the logic is much simpler if we just look at the length of continuation tokens
                ctxlen = len(context_toks) - max(
                    0,
                    len(context_toks) + len(continuation_toks) - (self.max_length + 1),
                )
                inps.append(inp)
                ctxlens.append(ctxlen)
                chunk_metadata.append(
                    (cache_key, context_toks, continuation_toks, ctxlen)
                )

            all_chunk_metadata.append(chunk_metadata)
            all_chunks.append((inps, {}))

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

        res = []
        for chunk_metadata, chunk_response in zip(all_chunk_metadata, responses):
            for (cache_key, context_enc, continuation_enc, ctxlen), response in zip(
                chunk_metadata, chunk_response.choices
            ):
                answer = get_result(response, ctxlen)
                res.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        if not requests:
            return []
        res = defaultdict(list)
        re_ords = {}
        all_chunks = []
        all_chunk_metadata = []

        grouper = lm_eval.models.utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            re_ord = utils.Reorderer(
                [req.args for req in reqs], lambda x: (-len(x[0]), x[0])
            )
            re_ords[key] = re_ord

            chunks = list(
                lm_eval.models.utils.chunks(re_ord.get_reordered(), n=self.batch_size)
            )
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                gen_kwargs = all_gen_kwargs[0]
                until = None
                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [until]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                            )
                        kwargs["stop"] = until
                    kwargs["max_tokens"] = kwargs.pop("max_gen_toks", self.max_gen_toks)
                    # OpenAI-style APIs use temperature, not do_sample; mirror HF's
                    # greedy default unless the caller samples or sets temperature.
                    if kwargs.pop("do_sample", None) is not True:
                        kwargs.setdefault("temperature", 0.0)
                    if self.add_bos:
                        kwargs["extra_body"] = {
                            **kwargs.get("extra_body", {}),
                            "add_BOS": True,
                        }
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )
                all_chunks.append((contexts, kwargs))
                all_chunk_metadata.append((key, chunk, until))

        responses = self.load_balancer.submit_requests(
            requests=all_chunks,
            total_requests=len(requests),
            disable_tqdm=disable_tqdm,
            cache_hook=self.cache_hook,
            model=self.model,
        )

        for (key, chunk, until), response in zip(all_chunk_metadata, responses):
            chunk_responses = []
            for resp, (context, args_) in zip(response.choices, chunk):
                s = resp.text
                if until is not None:
                    for term in until:
                        if len(term) > 0:
                            s = s.split(term)[0]
                chunk_responses.append(s)
                self.cache_hook.add_partial("generate_until", (context, args_), s)
            res[key].extend(chunk_responses)

        for key in res.keys():
            res[key] = re_ords[key].get_original(res[key])

        return grouper.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> List[float]:

        ll_token_requests: list[LLTokenRequest] = []
        num_windows_per_string: list[int] = []
        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows: list[tuple[list[int], list[int]]] = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            ll_token_requests.extend([(None,) + x for x in rolling_token_windows])
            num_windows_per_string.append(len(rolling_token_windows))

        window_nlls = self._loglikelihood_tokens(ll_token_requests, disable_tqdm=True)
        window_nlls = [x[0] for x in window_nlls]  # discard is_greedy
        loglikelihoods, cur = [], 0
        for num_windows in num_windows_per_string:
            string_nll = sum(window_nlls[cur : cur + num_windows])
            cur += num_windows
            loglikelihoods.append(string_nll)
        return loglikelihoods


@register_model("openai-chat-completions", "local-chat-completions")
class OpenaiChatCompletionsLM(LM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",  # GPT model or Local model using HuggingFace model paths
        base_url: str = None,
        truncate: bool = False,
        **kwargs,
    ) -> None:
        """

        :param model: str
            Implements an OpenAI-style chat completion API for
            accessing both OpenAI OR locally-hosted models using
            HuggingFace Tokenizer
            OpenAI API model (e.g. gpt-3.5-turbo)
            using the **gen_kwargs passed on init
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        try:
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = base_url
        self.truncate = truncate
        max_concurrent_per_client = int(kwargs.pop("max_concurrent_per_client", 1))
        self.load_balancer = LoadBalancedEndpoints(base_url, max_concurrent_per_client)

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        res = defaultdict(list)
        re_ords = {}
        all_chunks = []
        all_chunk_metadata = []

        grouper = lm_eval.models.utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            re_ord = utils.Reorderer(
                [req.args for req in reqs], lambda x: (-len(x[0]), x[0])
            )
            re_ords[key] = re_ord

            chunks = list(lm_eval.models.utils.chunks(re_ord.get_reordered(), n=1))
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                inps = [{"role": "user", "content": context} for context in contexts]
                gen_kwargs = all_gen_kwargs[0]
                until = None
                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [until]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                            )
                        kwargs["stop"] = until
                    kwargs["max_tokens"] = kwargs.pop("max_gen_toks", self.max_gen_toks)
                    # OpenAI-style APIs use temperature, not do_sample; mirror HF's
                    # greedy default unless the caller samples or sets temperature.
                    if kwargs.pop("do_sample", None) is not True:
                        kwargs.setdefault("temperature", 0.0)
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )
                all_chunks.append((inps, kwargs))
                all_chunk_metadata.append((key, chunk, until))

        responses = self.load_balancer.submit_requests(
            requests=all_chunks,
            total_requests=len(requests),
            disable_tqdm=disable_tqdm,
            cache_hook=self.cache_hook,
            model=self.model,
        )

        for (key, chunk, until), response in zip(all_chunk_metadata, responses):
            chunk_responses = []
            for resp, (context, args_) in zip(response.choices, chunk):
                s = resp.message.content
                if until is not None:
                    for term in until:
                        if len(term) > 0:
                            s = s.split(term)[0]
                chunk_responses.append(s)
                self.cache_hook.add_partial("generate_until", (context, args_), s)
            res[key].extend(chunk_responses)

        for key in res.keys():
            res[key] = re_ords[key].get_original(res[key])

        return grouper.get_original(res)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")
