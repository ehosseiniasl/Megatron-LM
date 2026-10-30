"""Tests for Surge-MM packed JSONL/tar loading."""

from __future__ import annotations
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from orchestration.perplexity_datasets.surge_mm import (
    SurgeMMDatasetError,
    SurgeMMImageReader,
    load_packed_jsonl_tar,
)


def write_packed_fixture(root: Path, records: list[dict[str, object]]) -> Path:
    data = root / "data.jsonl"
    data.write_text("".join(json.dumps(record) + "\n" for record in records))
    with tarfile.open(root / "data.tar", "w") as outer:
        for record in records:
            uid = str(record["uid"])
            nested_bytes = io.BytesIO()
            with tarfile.open(fileobj=nested_bytes, mode="w") as nested:
                image = f"image-{uid}".encode()
                info = tarfile.TarInfo(uid)
                info.size = len(image)
                nested.addfile(info, io.BytesIO(image))
            payload = nested_bytes.getvalue()
            info = tarfile.TarInfo(uid)
            info.size = len(payload)
            outer.addfile(info, io.BytesIO(payload))
    return data


def record(uid: str, description: str = "a description") -> dict[str, object]:
    return {"uid": uid, "tags": ["image_type_chart"], "data": {"answer": [description]}}


class SurgeMMDatasetTest(unittest.TestCase):
    def test_loads_ordered_metadata_and_exact_nested_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = write_packed_fixture(Path(temporary), [record("a"), record("b")])
            samples = load_packed_jsonl_tar(data)
            self.assertEqual([sample.uid for sample in samples], ["a", "b"])
            self.assertEqual(samples[0].category, "chart")
            with SurgeMMImageReader() as reader:
                self.assertEqual(reader.read(samples[1]), b"image-b")
            self.assertEqual([sample.uid for sample in load_packed_jsonl_tar(data, limit=1)], ["a"])

    def test_rejects_duplicate_ids_and_missing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = write_packed_fixture(root, [record("same"), record("same")])
            with self.assertRaisesRegex(SurgeMMDatasetError, "duplicate"):
                load_packed_jsonl_tar(data)
            (root / "data.tar").unlink()
            with self.assertRaisesRegex(SurgeMMDatasetError, "archive is missing"):
                load_packed_jsonl_tar(data)


if __name__ == "__main__":
    unittest.main()
