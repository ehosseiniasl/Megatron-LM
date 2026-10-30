from dataclasses import dataclass, field
from typing import Literal, Optional, Any, Union, Tuple, NamedTuple


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]

LLRollingArg = Tuple[str]
LLArg = Tuple[str, str]
GenUntilArg = Tuple[str, dict[str, Any]]
Arg = Union[LLArg, LLRollingArg, GenUntilArg]

LLTokenRequest = NamedTuple("LLTokenRequest", [("cache_key", str), ("context_tokens", list[int]), ("continuation_tokens", list[int])])

@dataclass
class Instance:
    request_type: OutputType
    doc: dict
    arguments: Arg
    idx: int
    metadata: Tuple[Optional[str], Optional[int], Optional[int]] = field(
        default_factory=lambda: (None, None, None)
    )
    token_window: Optional[int] = None
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: Optional[str] = None
    doc_id: Optional[int] = None
    repeats: Optional[int] = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata

    @property
    def args(self) -> Arg:
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        assert isinstance(self.arguments, tuple), f"Expected tuple, got {type(self.arguments)}"
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )
