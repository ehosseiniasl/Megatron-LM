"""Load nvbench-owned predefined few-shot demonstrations."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path


class PredefinedDemonstrationError(ValueError):
    """Raised when a dataset cannot provide the requested predefined examples."""


_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "predefined_demonstrations.json"
_ASSETS = _ROOT / "predefined_demonstration_assets"
_DATASET_FAMILIES = ("MMMU_Pro", "RealWorldQA", "DocVQA", "OCRBench", "MMMU")


def _dataset_family(dataset: str) -> str:
    if type(dataset) is not str or not dataset:
        raise PredefinedDemonstrationError("dataset must be nonempty text")
    for family in _DATASET_FAMILIES:
        if dataset == family or dataset.startswith((f"{family}_", f"{family}-")):
            return family
    raise PredefinedDemonstrationError(
        f"dataset {dataset!r} has no predefined demonstrations"
    )


@lru_cache(maxsize=1)
def _payload() -> Mapping[str, object]:
    try:
        value = json.loads(_DATA.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PredefinedDemonstrationError(
            "predefined demonstration data is unavailable"
        ) from error
    if not isinstance(value, dict):
        raise PredefinedDemonstrationError("predefined demonstration data must be a mapping")
    return value


def _records(dataset: str, style: str) -> Sequence[object]:
    family = _dataset_family(dataset)
    family_styles = _payload().get(family)
    if not isinstance(family_styles, dict):
        raise PredefinedDemonstrationError(
            f"dataset {dataset!r} has no predefined demonstrations"
        )
    records = family_styles.get(style)
    if not isinstance(records, list):
        raise PredefinedDemonstrationError(
            f"dataset {dataset!r} does not support {style!r} demonstrations"
        )
    return records


def load_predefined_demonstrations(
    dataset: str, style: str, num_shot: int
) -> tuple[tuple[str, dict[str, object]], ...]:
    """Return isolated predefined records and stable source IDs.

    ``style`` must be ``rich``, ``trivial``, or ``reasoning``. Images resolve
    only beneath the checked-in asset directory. Unsupported combinations,
    excessive counts, malformed records, or missing assets fail before prompt
    construction.
    """
    if style not in {"reasoning", "rich", "trivial"}:
        raise PredefinedDemonstrationError(
            "predefined style must be reasoning, rich, or trivial"
        )
    if type(num_shot) is not int or num_shot < 0:
        raise PredefinedDemonstrationError("num_shot must be a nonnegative integer")
    records = _records(dataset, style)
    if num_shot == 0:
        return ()
    if num_shot > len(records):
        raise PredefinedDemonstrationError(
            f"dataset {dataset!r} provides only {len(records)} {style!r} demonstrations"
        )

    family = _dataset_family(dataset)
    selected: list[tuple[str, dict[str, object]]] = []
    for position, raw in enumerate(records[:num_shot]):
        if not isinstance(raw, dict):
            raise PredefinedDemonstrationError("predefined demonstration must be a mapping")
        record = copy.deepcopy(raw)
        source_index = record.get("index")
        image_file = record.pop("image_file", None)
        if type(source_index) is not str or not source_index:
            raise PredefinedDemonstrationError("predefined demonstration has no stable index")
        if type(image_file) is not str or not image_file:
            raise PredefinedDemonstrationError("predefined demonstration has no image asset")
        try:
            image_path = (_ASSETS / image_file).resolve(strict=True)
        except OSError as error:
            raise PredefinedDemonstrationError(
                f"predefined demonstration image is unavailable: {image_file}"
            ) from error
        if not image_path.is_relative_to(_ASSETS) or not image_path.is_file():
            raise PredefinedDemonstrationError("predefined demonstration image escapes assets")
        record["image_path"] = str(image_path)
        selected.append((f"nvbench:{family}:{style}:{position}:{source_index}", record))
    return tuple(selected)


__all__ = ("PredefinedDemonstrationError", "load_predefined_demonstrations")
