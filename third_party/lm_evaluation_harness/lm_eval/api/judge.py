import asyncio
import inspect
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


DEFAULT_JUDGE_CONCURRENCY = 128
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_INFERENCE_BASE_URL = "https://integrate.api.nvidia.com/v1"

ParsedT = TypeVar("ParsedT")
MetadataT = TypeVar("MetadataT")


@dataclass(frozen=True)
class JudgeProvider:
    base_url: str
    api_key_envs: tuple[str, ...]
    supports_response_format: bool = True
    default_headers: dict[str, str] = field(default_factory=dict)


JUDGE_PROVIDERS = {
    "openrouter": JudgeProvider(
        base_url=OPENROUTER_BASE_URL,
        api_key_envs=("OPENROUTER_API_KEY",),
        default_headers={
            "HTTP-Referer": "https://github.com/EleutherAI/lm-evaluation-harness",
            "X-Title": "lm-evaluation-harness judge",
        },
    ),
    "nvidia": JudgeProvider(
        base_url=NVIDIA_INFERENCE_BASE_URL,
        api_key_envs=("NVIDIA_API_KEY", "NVIDIA_INFERENCE_API_KEY"),
        supports_response_format=False,
    ),
}


def get_judge_provider(provider: str) -> JudgeProvider:
    try:
        return JUDGE_PROVIDERS[provider.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(JUDGE_PROVIDERS))
        raise ValueError(
            f"Unsupported judge provider {provider!r}; expected one of {supported}"
        ) from error


def judge_provider_credential_help() -> str:
    return (
        "Available judge providers and credentials:\n"
        "  openrouter: set OPENROUTER_API_KEY\n"
        "  nvidia: set NVIDIA_API_KEY or NVIDIA_INFERENCE_API_KEY\n"
        "For Parametric QA, select one with "
        "PARAMETRIC_QA_JUDGE_PROVIDER=openrouter|nvidia. "
        "For the standalone grader, use --provider openrouter|nvidia."
    )


@dataclass(frozen=True)
class JudgeRequest(Generic[MetadataT]):
    messages: list[dict[str, Any]]
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    max_tokens: int = 1024
    temperature: float | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: MetadataT | None = None


@dataclass(frozen=True)
class JudgeResult(Generic[ParsedT, MetadataT]):
    request: JudgeRequest[MetadataT]
    value: ParsedT | None
    raw_response: str
    usage: dict[str, int | float]
    error: str | None = None


class AsyncOpenAIJudge:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: str | None = None,
        api_key_env: str | tuple[str, ...] = "OPENROUTER_API_KEY",
        concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
        timeout: float = 120.0,
        max_retries: int = 6,
        response_retries: int = 2,
        supports_response_format: bool = True,
        default_headers: dict[str, str] | None = None,
    ):
        if concurrency < 1:
            raise ValueError("Judge concurrency must be positive")
        if response_retries < 0:
            raise ValueError("Judge response retries cannot be negative")

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.response_retries = response_retries
        self.supports_response_format = supports_response_format
        self.default_headers = default_headers

    @staticmethod
    def _usage_values(response) -> dict[str, int | float]:
        usage = {}
        if response.usage is not None:
            for field_name in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
            ):
                value = getattr(response.usage, field_name, None)
                if value is not None:
                    usage[field_name] = value
        return usage

    async def _run_one(
        self,
        client,
        semaphore: asyncio.Semaphore,
        request: JudgeRequest[MetadataT],
        parse_response: Callable[[str], ParsedT | None],
    ) -> JudgeResult[ParsedT, MetadataT]:
        total_usage: dict[str, int | float] = {}
        raw_response = ""
        try:
            for attempt in range(self.response_retries + 1):
                kwargs = {
                    "model": self.model,
                    "messages": request.messages,
                    "max_tokens": request.max_tokens,
                    **request.extra_kwargs,
                }
                if (
                    self.supports_response_format
                    and request.response_format is not None
                ):
                    kwargs["response_format"] = request.response_format
                if request.seed is not None:
                    kwargs["seed"] = request.seed + attempt * 1_000_003
                if request.temperature is not None:
                    kwargs["temperature"] = request.temperature

                async with semaphore:
                    response = await client.chat.completions.create(**kwargs)

                raw_response = response.choices[0].message.content or ""
                for key, value in self._usage_values(response).items():
                    total_usage[key] = total_usage.get(key, 0) + value
                parsed = parse_response(raw_response)
                if parsed is not None:
                    return JudgeResult(
                        request=request,
                        value=parsed,
                        raw_response=raw_response,
                        usage=total_usage,
                    )
            error = (
                f"Unparsable judge response after {self.response_retries + 1} "
                f"attempts: {raw_response!r}"
            )
        except Exception as exception:
            error = str(exception)

        return JudgeResult(
            request=request,
            value=None,
            raw_response=raw_response,
            usage=total_usage,
            error=error,
        )

    async def run(
        self,
        requests: Iterable[JudgeRequest[MetadataT]],
        parse_response: Callable[[str], ParsedT | None],
        on_result: Callable[[JudgeResult[ParsedT, MetadataT]], Any] | None = None,
    ) -> list[JudgeResult[ParsedT, MetadataT]]:
        requests = list(requests)
        if not requests:
            return []

        api_key_envs = (
            (self.api_key_env,)
            if isinstance(self.api_key_env, str)
            else self.api_key_env
        )
        api_key = self.api_key or next(
            (
                os.environ.get(env_name)
                for env_name in api_key_envs
                if os.environ.get(env_name)
            ),
            None,
        )
        if not api_key:
            raise RuntimeError(
                "JUDGE_API_KEY_MISSING: no API key was found; checked "
                f"{' or '.join(api_key_envs)}.\n{judge_provider_credential_help()}"
            )

        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
        )
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_indexed(index, request):
            result = await self._run_one(
                client,
                semaphore,
                request,
                parse_response,
            )
            return index, result

        indexed_jobs = [
            asyncio.create_task(run_indexed(index, request))
            for index, request in enumerate(requests)
        ]
        results: list[JudgeResult[ParsedT, MetadataT] | None] = [None] * len(requests)
        try:
            for job in asyncio.as_completed(indexed_jobs):
                index, result = await job
                results[index] = result
                if on_result is not None:
                    callback_result = on_result(result)
                    if inspect.isawaitable(callback_result):
                        await callback_result
        finally:
            await client.close()

        return [result for result in results if result is not None]
