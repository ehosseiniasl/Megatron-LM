import contextlib
import io
import tarfile
import zlib
from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Literal, Optional

from mistral_fork.shared.common_utils import TarOffset
from mistral_fork.shared.s3_utils import DEFAULT_BACKEND, Backend, get_object_content_from_bucket

TAR_HEADER_SIZE = 512


class Tar:
    def __init__(self, tar_obj: tarfile.TarFile) -> None:
        self._tar = tar_obj
        self._open = True
        self.names: set[str] = set()

    @property
    def n_elements(self) -> int:
        return len(self.names)

    def _add_name(self, name: str) -> None:
        assert self._open
        if name in self.names:
            raise ValueError(f"{name} is already in the tar")
        self.names.add(name)

    def _write(self, data: bytes, name: str, compress: bool) -> TarOffset:
        assert self._open
        assert self._tar.fileobj is not None
        assert len(name.encode("utf-8")) <= 100
        self._add_name(name)

        if compress:
            import zstandard as zstd

            data = zstd.compress(data)

        tar_info = tarfile.TarInfo(name=name)
        tar_info.size = len(data)

        offset_start = self._tar.fileobj.tell()
        with io.BytesIO(data) as bio:
            self._tar.addfile(tarinfo=tar_info, fileobj=bio)
        offset_end = self._tar.fileobj.tell()

        length = offset_end - offset_start
        return TarOffset(offset=offset_start, size=length, new_tar_format=True)

    def _write_file(self, path: Path, compress: bool) -> TarOffset:
        return self._write(data=path.read_bytes(), name=path.name, compress=compress)

    def _write_bytes(self, data: bytes, name: str, compress: bool) -> TarOffset:
        return self._write(data=data, name=name, compress=compress)

    @abstractmethod
    def write_file(self, path: Path) -> TarOffset:
        raise NotImplementedError

    @abstractmethod
    def write_bytes(self, data: bytes, name: str) -> TarOffset:
        raise NotImplementedError

    def close(self) -> None:
        assert self._open is True
        self._tar.close()
        self._open = False


class MemoryTar(Tar):
    def __init__(self) -> None:
        self._buffer = io.BytesIO()
        tar = tarfile.open(fileobj=self._buffer, mode="w")
        super().__init__(tar)

    def write_file(self, path: Path) -> TarOffset:
        return self._write_file(path=path, compress=False)

    def write_bytes(self, data: bytes, name: str) -> TarOffset:
        return self._write_bytes(data=data, name=name, compress=False)

    def get_bytes(self) -> bytes:
        self.close()
        return self._buffer.getvalue()

    def export(self, path: Path) -> None:
        with path.open("wb") as f:
            f.write(self.get_bytes())


class DiskTar(Tar):
    def __init__(
        self,
        tar_path: Path,
        compress: bool,
        open_mode: Literal["w|", "a"] = "w|",
    ) -> None:
        assert open_mode in ("w|", "a")
        tar = tarfile.open(str(tar_path), mode=open_mode)
        self.compress = compress
        super().__init__(tar)

    def write_file(self, path: Path) -> TarOffset:
        return self._write_file(path=path, compress=self.compress)

    def write_bytes(self, data: bytes, name: str) -> TarOffset:
        return self._write_bytes(data=data, name=name, compress=self.compress)

    @staticmethod
    @contextlib.contextmanager
    def open(
        tar_path: Path,
        compress: bool = False,
        open_mode: Literal["w|", "a"] = "w|",
    ) -> Iterator["DiskTar"]:
        assert isinstance(compress, bool), compress
        instance = DiskTar(tar_path, compress=compress, open_mode=open_mode)
        try:
            yield instance
        finally:
            instance.close()


def _read_bytes(is_compressed: bool, bytes: bytes) -> bytes:
    if is_compressed:
        return zlib.decompress(bytes)
    return bytes


def _load_bytes_from_path(tar_path: Path, offset: int, size: int) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(offset)
        result = f.read(size)
    if len(result) != size:
        raise OSError(
            f"Expected {size} bytes from {tar_path} at offset {offset}, "
            f"received {len(result)}"
        )
    return result


def load_bytes(tar_path: str, offset: int, size: int) -> bytes:
    """Load raw bytes from a tar file on disk or S3."""
    if tar_path.startswith("s3://"):
        return _load_from_s3(tar_path, offset, size)
    return _load_bytes_from_path(Path(tar_path), offset, size)


def _load_from_s3(
    tar_path: str,
    offset: int,
    size: int,
    backend: Backend = DEFAULT_BACKEND,
) -> bytes:
    bucket_name, tar_name = tar_path[5:].split("/", 1)
    return get_object_content_from_bucket(
        tar_name,
        bucket_name,
        offset=offset,
        size=size,
        backend=backend,
    )


def get_tar_bytes_helper(
    tar_path: str,
    offset: TarOffset,
    is_compressed: bool,
    read_full_tar: bool = False,
    backend: Backend = DEFAULT_BACKEND,
) -> Optional[bytes]:
    if offset.size == 0:
        return None

    if tar_path.endswith(".tar.gz"):
        raise ValueError(
            "Tar.gz files are not supported. First decompress them and then share the path to the tar file."
        )

    if offset.new_tar_format:
        # New TAR format: skip header, read content
        content_offset = offset.offset + TAR_HEADER_SIZE
        content_size = offset.size - TAR_HEADER_SIZE
        if tar_path.startswith("s3://"):
            _bytes = _load_from_s3(tar_path, content_offset, content_size, backend)
        else:
            _bytes = _load_bytes_from_path(Path(tar_path), content_offset, content_size)
        return _read_bytes(is_compressed=is_compressed, bytes=_bytes)
    else:
        # Old TAR format
        if tar_path.startswith("s3://"):
            _bytes = _load_from_s3(tar_path, offset.offset, offset.size, backend)
        else:
            _bytes = _load_bytes_from_path(Path(tar_path), offset.offset, offset.size)
        return _read_bytes(is_compressed=is_compressed, bytes=_bytes)
