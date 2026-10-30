"""Read Surge-MM conditional-perplexity metadata and packed images."""

from __future__ import annotations

import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from ..utils import StrictJsonError, load_json_bytes


class SurgeMMDatasetError(RuntimeError):
    """Raised when Surge-MM metadata or packed image bytes are invalid."""


@dataclass(frozen=True)
class SurgeMMSample:
    """One packed image and reference description to score."""

    uid: str
    description: str
    category: str
    image_member: str
    image_archive: Path


_MAXIMUM_RECORD_BYTES = 32 * 1024 * 1024


def _nonempty_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise SurgeMMDatasetError(f"{label} must be a nonempty string")
    return value.strip()


def load_packed_jsonl_tar(
    data_path: Path, *, limit: int | None = None
) -> tuple[SurgeMMSample, ...]:
    """Load ordered samples from JSONL metadata and its sibling nested tar."""
    if not isinstance(data_path, Path) or not data_path.is_file():
        raise SurgeMMDatasetError("packed JSONL path must be a readable file")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise SurgeMMDatasetError("limit must be a positive integer")
    archive = data_path.with_suffix(".tar")
    if not archive.is_file():
        raise SurgeMMDatasetError(f"packed image archive is missing: {archive}")

    samples: list[SurgeMMSample] = []
    seen: set[str] = set()
    try:
        with data_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                try:
                    record = load_json_bytes(raw_line, maximum_bytes=_MAXIMUM_RECORD_BYTES)
                except StrictJsonError as error:
                    raise SurgeMMDatasetError(
                        f"invalid JSONL record at line {line_number}"
                    ) from error
                if type(record) is not dict:
                    raise SurgeMMDatasetError(f"JSONL record {line_number} must be an object")
                uid = _nonempty_text(record.get("uid"), f"record {line_number} uid")
                if uid in seen:
                    raise SurgeMMDatasetError(f"duplicate sample uid: {uid}")
                data = record.get("data")
                tags = record.get("tags")
                if type(data) is not dict or type(tags) is not list or not tags:
                    raise SurgeMMDatasetError(f"record {line_number} is missing data or tags")
                answers = data.get("answer")
                if type(answers) is not list or not answers:
                    raise SurgeMMDatasetError(
                        f"record {line_number} is missing its reference answer"
                    )
                tag = _nonempty_text(tags[0], f"record {line_number} image type")
                samples.append(
                    SurgeMMSample(
                        uid=uid,
                        description=_nonempty_text(
                            answers[0], f"record {line_number} reference answer"
                        ),
                        category=tag.removeprefix("image_type_"),
                        image_member=uid,
                        image_archive=archive.resolve(strict=True),
                    )
                )
                seen.add(uid)
                if limit is not None and len(samples) >= limit:
                    break
    except OSError as error:
        raise SurgeMMDatasetError("cannot read packed JSONL data") from error
    if not samples:
        raise SurgeMMDatasetError("packed JSONL data contains no usable samples")
    return tuple(samples)


class SurgeMMImageReader:
    """Read packed images with per-thread archive caches and explicit cleanup."""

    def __init__(self) -> None:
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._open_handles: list[tarfile.TarFile] = []
        self._closed = False

    def __enter__(self) -> SurgeMMImageReader:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close every archive opened by a worker thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            handles = tuple(self._open_handles)
            self._open_handles.clear()
        for handle in handles:
            handle.close()

    def read(self, sample: SurgeMMSample) -> bytes:
        """Read a same-named image from an outer member containing a nested tar."""
        handles = getattr(self._thread_local, "tar_handles", None)
        if handles is None:
            handles = {}
            self._thread_local.tar_handles = handles
        outer = handles.get(sample.image_archive)
        if outer is None:
            outer = self._open_archive(sample.image_archive)
            handles[sample.image_archive] = outer
        try:
            nested_file = outer.extractfile(sample.image_member)
            if nested_file is None:
                raise KeyError(sample.image_member)
            with nested_file, tarfile.open(fileobj=nested_file) as nested:
                image_file = nested.extractfile(sample.image_member)
                if image_file is None:
                    raise KeyError(sample.image_member)
                with image_file:
                    return image_file.read()
        except (KeyError, OSError, tarfile.TarError) as error:
            raise SurgeMMDatasetError(
                f"cannot read packed image for sample {sample.uid}"
            ) from error

    def _open_archive(self, path: Path) -> tarfile.TarFile:
        with self._lock:
            if self._closed:
                raise SurgeMMDatasetError("packed image reader is closed")
            try:
                handle = tarfile.open(path)
            except (OSError, tarfile.TarError) as error:
                raise SurgeMMDatasetError("cannot open packed image archive") from error
            self._open_handles.append(handle)
            return handle


__all__ = (
    "SurgeMMDatasetError",
    "SurgeMMImageReader",
    "SurgeMMSample",
    "load_packed_jsonl_tar",
)
