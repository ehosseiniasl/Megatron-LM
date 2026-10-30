"""Build few-shot conversations and normalize their demonstration answers."""

from __future__ import annotations

import ast
import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol

from ..utils import VLMEvalRowIdentityError, VLMEvalRowIndex, require_vlmeval_row_index
from .prompt_adapters import apply_prompt_adapter

if TYPE_CHECKING:
    from ..configured_eval_run import BenchmarkConfig
    from .demonstration_selection import DemonstrationExample, FewshotRowSelection


class FewshotConversationError(ValueError):
    """Raised when a demonstration cannot produce an unambiguous answer turn."""


class VLMEvalMessageError(ValueError):
    """Raised when an upstream prompt does not satisfy the flat item contract."""


class DemonstrationAnswerNormalizationError(ValueError):
    """Raised when a benchmark value cannot produce one answer-text scalar."""


class _PromptDataset(Protocol):
    def build_prompt(self, record: object) -> object:
        """Return native VLMEvalKit prompt items for one independent record."""
        ...


def _literal_sequence(value: str) -> Sequence[object] | None:
    if not any(
        value.startswith(left) and value.endswith(right) for left, right in (("[", "]"), ("(", ")"))
    ):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (MemoryError, RecursionError, SyntaxError, ValueError):
        return None
    return parsed if type(parsed) in (list, tuple) else None


def _normalize_answer_scalar(value: object) -> str:
    if type(value) is str:
        text = value.strip()
        if text:
            return text
    elif not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value):
        return str(value)
    raise DemonstrationAnswerNormalizationError(
        "demonstration answer must contain a nonempty text or finite numeric scalar"
    )


def normalize_demonstration_answer(value: object) -> str:
    """Return one deterministic answer string without executing input text.

    Native and safely parsed stringified lists/tuples use their first element,
    matching the historical VQA few-shot policy. Inputs are never mutated.
    Empty collections, nested first elements, booleans, non-finite numbers, and
    other non-scalars raise :class:`DemonstrationAnswerNormalizationError`.
    """
    sequence: Sequence[object] | None = None
    if type(value) in (list, tuple):
        sequence = value
    elif type(value) is str:
        stripped = value.strip()
        sequence = _literal_sequence(stripped)
        if sequence is None:
            return _normalize_answer_scalar(stripped)
    if sequence is not None:
        if not sequence:
            raise DemonstrationAnswerNormalizationError(
                "demonstration answer collection must not be empty"
            )
        return _normalize_answer_scalar(sequence[0])
    return _normalize_answer_scalar(value)


def copy_vlmeval_prompt_items(
    prompt: object, *, include_images: bool = True
) -> list[dict[str, str]]:
    """Return an isolated copy of validated native VLMEvalKit prompt items.

    ``prompt`` must be a nonempty sequence of exact ``type``/``value`` mappings.
    Native system-tagged text items are also accepted and flattened into the
    user turn because the contained transport deliberately supports alternating
    user/assistant turns only. Nonempty text is retained; empty text separators
    emitted by native dataset prompt splitters are omitted. Images are omitted
    only when ``include_images`` is false. Inputs are not mutated. Invalid items
    or an empty filtered result raise :class:`VLMEvalMessageError`.
    """
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes, bytearray)):
        raise VLMEvalMessageError("VLMEval prompt must be a sequence of item mappings")

    copied: list[dict[str, str]] = []
    for position, item in enumerate(prompt):
        if not isinstance(item, dict):
            raise VLMEvalMessageError(f"prompt item {position} must be a mapping")
        fields = set(item)
        plain = fields == {"type", "value"}
        system_text = fields == {"role", "type", "value"}
        if not plain and not system_text:
            raise VLMEvalMessageError(f"prompt item {position} has unsupported fields")
        item_type: Any = item["type"]
        value: Any = item["value"]
        if type(item_type) is not str or item_type not in {"text", "image"}:
            raise VLMEvalMessageError(f"prompt item {position} has unsupported type: {item_type!r}")
        if system_text and (item.get("role") != "system" or item_type != "text"):
            raise VLMEvalMessageError(f"prompt item {position} has an unsupported role")
        if not isinstance(value, str):
            raise VLMEvalMessageError(f"prompt item {position} value must be a string")
        if item_type == "text" and not value.strip():
            continue
        if item_type == "image" and not value.strip():
            raise VLMEvalMessageError(f"prompt item {position} image value must be non-empty")
        if item_type == "image" and not include_images:
            continue
        copied.append({"type": item_type, "value": value})

    if not copied:
        raise VLMEvalMessageError("VLMEval prompt has no content after image filtering")
    return copied


def _required_text(record: Mapping[str, object], field: str) -> str:
    if field not in record or record[field] is None:
        raise FewshotConversationError(f"demonstration is missing {field}")
    try:
        return normalize_demonstration_answer(record[field])
    except DemonstrationAnswerNormalizationError as error:
        raise FewshotConversationError(f"demonstration {field} is invalid") from error


def _options(record: Mapping[str, object]) -> dict[str, str]:
    options = {}
    for key, raw_value in record.items():
        if not isinstance(key, str):
            continue
        if len(key) != 1 or not key.isascii() or not key.isalpha() or not key.isupper():
            continue
        if raw_value is None:
            continue
        if isinstance(raw_value, Real) and not isinstance(raw_value, bool):
            if not math.isfinite(raw_value):
                continue
        value = str(raw_value).strip()
        if value:
            options[key] = value
    return options


def _label_answer(record: Mapping[str, object]) -> str:
    options = _options(record)
    if "answer_label" in record:
        label = _required_text(record, "answer_label")
        if label not in options:
            raise FewshotConversationError(f"answer label {label!r} has no matching option")
        return label

    answer = _required_text(record, "answer")
    if answer in options:
        return answer
    matches = [label for label, value in options.items() if value == answer]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FewshotConversationError("answer value matches more than one option")
    raise FewshotConversationError("answer cannot be resolved to an option label")


def _value_answer(record: Mapping[str, object]) -> str:
    if "answer_value" in record:
        return _required_text(record, "answer_value")
    answer = _required_text(record, "answer")
    options = _options(record)
    if answer in options:
        return options[answer]
    matches = [value for value in options.values() if value == answer]
    if len(matches) == 1 or not options:
        return answer
    if len(matches) > 1:
        raise FewshotConversationError("answer value matches more than one option")
    raise FewshotConversationError("answer cannot be resolved to an option value")


def _screenspot_click_center(record: Mapping[str, object]) -> tuple[float, float]:
    raw_bbox = record.get("bbox")
    bbox: object = raw_bbox
    if type(raw_bbox) is str:
        bbox = _literal_sequence(raw_bbox.strip())
    if type(bbox) not in (list, tuple) or len(bbox) != 4:
        raise FewshotConversationError(
            "ScreenSpot demonstration bbox must contain x, y, width, and height"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in bbox
    ):
        raise FewshotConversationError(
            "ScreenSpot demonstration bbox values must be finite numbers"
        )
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        raise FewshotConversationError(
            "ScreenSpot demonstration bbox must have positive dimensions"
        )
    center_x = x + (width - 1) / 2
    center_y = y + (height - 1) / 2
    if center_x < 0 or center_y < 0:
        raise FewshotConversationError("ScreenSpot demonstration click must be nonnegative")

    return center_x, center_y


def _screenspot_click_answer(record: Mapping[str, object]) -> str:
    center_x, center_y = _screenspot_click_center(record)

    def coordinate(value: float) -> str:
        return str(int(value)) if value.is_integer() else format(value, ".12g")

    return f"pyautogui.click(x={coordinate(center_x)}, " f"y={coordinate(center_y)})"


def format_demonstration_answer(
    record: Mapping[str, object], answer_format: str, *, image_size: tuple[int, int] | None = None
) -> str:
    """Return one normalized answer turn from a read-only demonstration record.

    ``answer_format`` may be ``raw``, ``label``, ``value``, or
    ``screenspot_click`` or ``screenspot_click_normalized``. The normalized
    variant requires the original demo image's positive integer width/height
    in ``image_size``; it never uses resized inference dimensions. Historical
    stringified multi-annotator answers are
    safely reduced to their first scalar. ScreenSpot boxes become deterministic
    center-click commands without executing record text. Invalid or ambiguous
    values raise :class:`FewshotConversationError`.
    """
    if answer_format == "raw":
        return _required_text(record, "answer")
    if answer_format == "label":
        return _label_answer(record)
    if answer_format == "value":
        return _value_answer(record)
    if answer_format == "screenspot_click":
        return _screenspot_click_answer(record)
    if answer_format == "screenspot_click_normalized":
        center_x, center_y = _screenspot_click_center(record)
        if (
            type(image_size) is not tuple
            or len(image_size) != 2
            or any(type(value) is not int or value <= 0 for value in image_size)
        ):
            raise FewshotConversationError(
                "normalized ScreenSpot answer requires original image dimensions"
            )
        width, height = image_size
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise FewshotConversationError(
                "ScreenSpot demonstration center is outside its original image"
            )
        return f"pyautogui.click(x={center_x / width:.6f}, y={center_y / height:.6f})"
    raise FewshotConversationError(f"unsupported demonstration answer format: {answer_format}")


def _original_demo_image_size(prompt: Sequence[dict[str, str]]) -> tuple[int, int]:
    # Resolve the same native image used in the request, before image filtering
    # or transport resizing. Pillow stays in the contained client runtime.
    from PIL import Image

    images = [item["value"] for item in prompt if item["type"] == "image"]
    if len(images) != 1:
        raise FewshotConversationError(
            "normalized ScreenSpot demo requires exactly one original image"
        )
    try:
        with Image.open(images[0]) as image:
            return image.size
    except (OSError, ValueError) as error:
        raise FewshotConversationError(
            "cannot read original ScreenSpot demo image dimensions"
        ) from error


def build_fewshot_conversation(
    wrapped_dataset: _PromptDataset,
    demonstrations: Sequence[DemonstrationExample],
    query: object,
    *,
    include_demonstration_images: bool,
    demonstration_answer_format: str,
    prompt_adapter: str | None = None,
) -> list[dict[str, object]]:
    """Build a new alternating demonstration/query conversation.

    The wrapped dataset must provide ``build_prompt(record)`` in VLMEvalKit
    prompt-item format. Demonstration images may be omitted, but query images are
    always retained. Inputs are deep-copied before upstream prompt construction,
    and the returned message tree shares no mutable prompt items with them.
    Invalid owned answer/message contracts raise :class:`FewshotConversationError`
    or ``VLMEvalMessageError``; upstream ``build_prompt`` failures propagate.
    """
    if not callable(getattr(wrapped_dataset, "build_prompt", None)):
        raise FewshotConversationError("wrapped dataset must provide build_prompt")
    if type(include_demonstration_images) is not bool:
        raise FewshotConversationError("include_demonstration_images must be boolean")
    if type(demonstration_answer_format) is not str:
        raise FewshotConversationError("demonstration_answer_format must be text")
    messages: list[dict[str, object]] = []
    for position, demonstration in enumerate(demonstrations):
        record = getattr(demonstration, "record", None)
        if not isinstance(record, dict):
            raise FewshotConversationError(f"demonstration {position} record must be a dictionary")
        if demonstration_answer_format == "screenspot_click_normalized":
            prompt = copy_vlmeval_prompt_items(wrapped_dataset.build_prompt(copy.deepcopy(record)))
            answer = format_demonstration_answer(
                record, demonstration_answer_format, image_size=_original_demo_image_size(prompt)
            )
        else:
            answer = format_demonstration_answer(record, demonstration_answer_format)
            prompt = wrapped_dataset.build_prompt(copy.deepcopy(record))
        user_content = copy_vlmeval_prompt_items(
            prompt, include_images=include_demonstration_images
        )
        user_content = apply_prompt_adapter(prompt_adapter, user_content, record)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "value": answer}]})

    query_prompt = wrapped_dataset.build_prompt(copy.deepcopy(query))
    query_content = copy_vlmeval_prompt_items(query_prompt, include_images=True)
    query_content = apply_prompt_adapter(prompt_adapter, query_content, query)
    messages.append({"role": "user", "content": query_content})
    return messages


class VLMEvalFewshotDatasetProxyError(ValueError):
    """Raised when selected rows do not match the wrapped dataset."""


def _row_at(data: object, position: int) -> object:
    try:
        return data.iloc[position]  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise VLMEvalFewshotDatasetProxyError(
            f"cannot read scored row at position {position}"
        ) from error


def _record_copy(row: object) -> dict[str, object]:
    if isinstance(row, Mapping):
        return copy.deepcopy(dict(row))
    converter = getattr(row, "to_dict", None)
    if not callable(converter):
        raise VLMEvalFewshotDatasetProxyError("dataset row cannot be converted to a record")
    record = converter()
    if not isinstance(record, dict):
        raise VLMEvalFewshotDatasetProxyError("dataset row conversion did not return a dictionary")
    return copy.deepcopy(record)


def _copy_selected_data(data: object, positions: tuple[int, ...]) -> object:
    try:
        selected = data.iloc[list(positions)]  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise VLMEvalFewshotDatasetProxyError("cannot select scored dataset positions") from error
    copier = getattr(selected, "copy", None)
    if not callable(copier):
        return copy.deepcopy(selected)
    try:
        return copier(deep=True)
    except TypeError:
        return copier()


def _native_indices(data: object) -> tuple[VLMEvalRowIndex, ...]:
    try:
        values = tuple(data["index"])  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise VLMEvalFewshotDatasetProxyError(
            "scored dataset view must expose an index column"
        ) from error
    try:
        return tuple(
            require_vlmeval_row_index(value, label=f"scored index {position}")
            for position, value in enumerate(values)
        )
    except VLMEvalRowIdentityError as error:
        raise VLMEvalFewshotDatasetProxyError(str(error)) from error


def _validate_selection(
    data: object, selection: FewshotRowSelection, benchmark: BenchmarkConfig
) -> tuple[int, ...]:
    demonstrations = tuple(selection.demonstrations)
    demo_positions = tuple(selection.demonstration_positions)
    scored_positions = tuple(selection.scored_positions)
    if len(demonstrations) != benchmark.num_shot:
        raise VLMEvalFewshotDatasetProxyError(
            "demonstration count does not match benchmark num_shot"
        )
    source_ids = tuple(item.source_index for item in demonstrations)
    if any(type(value) is not str or not value for value in source_ids):
        raise VLMEvalFewshotDatasetProxyError("demonstration source IDs must be nonempty")
    if len(source_ids) != len(set(source_ids)):
        raise VLMEvalFewshotDatasetProxyError("demonstration source IDs must be unique")

    if benchmark.demonstration_source == "benchmark":
        if len(demo_positions) != benchmark.num_shot:
            raise VLMEvalFewshotDatasetProxyError(
                "benchmark demonstrations must identify their source positions"
            )
    elif demo_positions:
        raise VLMEvalFewshotDatasetProxyError(
            "packaged demonstrations cannot identify target-row positions"
        )

    all_positions = demo_positions + scored_positions
    if any(type(position) is not int or position < 0 for position in all_positions):
        raise VLMEvalFewshotDatasetProxyError("selected positions must be nonnegative integers")
    if len(demo_positions) != len(set(demo_positions)):
        raise VLMEvalFewshotDatasetProxyError("demonstration positions must be unique")
    if len(scored_positions) != len(set(scored_positions)):
        raise VLMEvalFewshotDatasetProxyError("scored positions must be unique")
    if set(demo_positions) & set(scored_positions):
        raise VLMEvalFewshotDatasetProxyError("demonstration and scored positions must be disjoint")
    try:
        row_count = len(data)  # type: ignore[arg-type]
    except TypeError as error:
        raise VLMEvalFewshotDatasetProxyError("wrapped dataset data must be sized") from error
    if all_positions and max(all_positions) >= row_count:
        raise VLMEvalFewshotDatasetProxyError("selected position exceeds dataset length")
    if benchmark.scored_limit is not None and len(scored_positions) > benchmark.scored_limit:
        raise VLMEvalFewshotDatasetProxyError("selection exceeds benchmark scored limit")

    if benchmark.demonstration_source == "benchmark":
        expected_ids = tuple(
            str(_record_copy(_row_at(data, position)).get("index")) for position in demo_positions
        )
        if source_ids != expected_ids:
            raise VLMEvalFewshotDatasetProxyError(
                "demonstration IDs do not match native source-row indices"
            )
    return scored_positions


class VLMEvalFewshotDatasetProxy:
    """Expose an isolated scored view while delegating upstream behavior.

    Construction validates the immutable partition against ``wrapped_dataset``
    and deep-copies only selected scored rows. It does not mutate the upstream
    dataset. Invalid datasets or partitions raise
    :class:`VLMEvalFewshotDatasetProxyError`.
    """

    force_use_dataset_prompt = True

    def __init__(
        self, wrapped_dataset: object, selection: FewshotRowSelection, benchmark: BenchmarkConfig
    ) -> None:
        if not callable(getattr(wrapped_dataset, "build_prompt", None)):
            raise VLMEvalFewshotDatasetProxyError("wrapped dataset has no build_prompt method")
        if not callable(getattr(wrapped_dataset, "evaluate", None)):
            raise VLMEvalFewshotDatasetProxyError("wrapped dataset has no evaluate method")
        source_data = getattr(wrapped_dataset, "data", None)
        if source_data is None:
            raise VLMEvalFewshotDatasetProxyError("wrapped dataset has no data")

        positions = _validate_selection(source_data, selection, benchmark)
        selected_data = _copy_selected_data(source_data, positions)
        selected_indices = _native_indices(selected_data)
        if selected_indices != tuple(selection.scored_indices):
            raise VLMEvalFewshotDatasetProxyError(
                "selection indices do not match native scored-row indices"
            )

        self._wrapped_dataset = wrapped_dataset
        self.selection = selection
        self.benchmark = benchmark
        self.data = selected_data

    @property
    def wrapped_dataset(self) -> object:
        """Return the original upstream dataset without transferring ownership."""
        return self._wrapped_dataset

    def __len__(self) -> int:
        return len(self.data)  # type: ignore[arg-type]

    def __getitem__(self, position: int) -> dict[str, object]:
        if type(position) is not int:
            raise TypeError("dataset position must be an integer")
        return _record_copy(_row_at(self.data, position))

    def build_prompt(self, query: object) -> list[dict[str, object]]:
        """Return a fresh few-shot conversation for one scored row or position.

        The wrapped dataset constructs all native prompt items; invalid rows or
        messages raise the relevant proxy or conversation error without mutation.
        """
        if type(query) is int:
            query = _row_at(self.data, query)
        return build_fewshot_conversation(
            self._wrapped_dataset,  # type: ignore[arg-type]
            self.selection.demonstrations,
            query,
            include_demonstration_images=self.benchmark.include_demonstration_images,
            demonstration_answer_format=self.benchmark.demonstration_answer_format,
            prompt_adapter=self.benchmark.prompt_adapter,
        )

    def evaluate(self, eval_file: object, **judge_kwargs: object) -> object:
        """Delegate ``eval_file`` and all judge keyword arguments unchanged.

        This mutates only as the upstream dataset's ``evaluate`` implementation
        specifies. Upstream exceptions propagate without wrapping.
        """
        return self._wrapped_dataset.evaluate(  # type: ignore[attr-defined]
            eval_file, **judge_kwargs
        )

    def __getattr__(self, name: str) -> Any:
        wrapped = object.__getattribute__(self, "_wrapped_dataset")
        return getattr(wrapped, name)
