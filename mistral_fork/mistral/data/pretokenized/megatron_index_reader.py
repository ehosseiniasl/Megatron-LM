# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Essentially re-written in entirety

"""
Part of this code was taken from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py
"""

import logging
import os
import shutil
import struct
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from itertools import accumulate
from types import TracebackType
from typing import Any, Optional, Union, assert_never, overload

import numpy
import torch


def log_single_rank(
    logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any
) -> None:
    """If torch distributed is initialized, log only on rank

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        rank (int, optional): The rank to write on. Defaults to 0.

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            logger.log(*args, **kwargs)
    else:
        logger.log(*args, **kwargs)


logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: type[numpy.number]) -> int:
        """Get the code from the dtype

        Args:
            value (Type[numpy.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> type[numpy.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            Type[numpy.number]: The dtype
        """
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, type[numpy.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, Type[numpy.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> type[numpy.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            Type[numpy.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


class _IndexWriter(object):
    """Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (Type[numpy.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()
        return None

    def write(self, sequence_lengths: numpy.ndarray) -> None:
        """Write the index (.idx) file."""
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        self.idx_writer.write(
            numpy.asarray(sequence_lengths, dtype=numpy.int32).tobytes(order="C")
        )
        self.idx_writer.write(
            numpy.asarray(sequence_pointers, dtype=numpy.int64).tobytes(order="C")
        )

    def _sequence_pointers(self, sequence_lengths: numpy.ndarray) -> numpy.ndarray:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (numpy.ndarray): The length of each sequence

        Returns:
            numpy.ndarray: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        return (
            numpy.concatenate(
                [[0], numpy.cumsum(sequence_lengths, dtype=numpy.int64)[:-1]]
            )
            * itemsize
            if len(sequence_lengths) > 0
            else numpy.empty(0, dtype=numpy.int64)
        )


class _IndexReader(object):
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file
    """

    def __init__(self, idx_path: str, multimodal: bool | None = None) -> None:
        log_single_rank(
            logger, logging.DEBUG, f"Load the {type(self).__name__} from {idx_path}"
        )

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = int(struct.unpack("<Q", stream.read(8))[0])
            self.document_count = int(struct.unpack("<Q", stream.read(8))[0])

            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer: None | memoryview = memoryview(self.bin_buffer_mmap)  # type: ignore

        log_single_rank(logger, logging.DEBUG, "\tExtract the sequence lengths")
        t_beg = time.time()
        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        t_end = time.time()
        log_single_rank(
            logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds"
        )

        log_single_rank(logger, logging.DEBUG, "\tExtract the sequence pointers")
        t_beg = time.time()
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        t_end = time.time()
        log_single_rank(
            logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds"
        )

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count

        log_single_rank(
            logger, logging.DEBUG, f"> total number of sequences: {len(self)}"
        )

        # Make sure we can pickel the object
        self.bin_buffer = None

    def __del__(self) -> None:
        """Clean up the object"""
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()  # type: ignore
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> tuple[numpy.int64, numpy.int32]:
        """Return the pointer and length at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.int64, numpy.int32]: The pointer and length at the index
        """
        if self.bin_buffer is None:
            self.bin_buffer = memoryview(self.bin_buffer_mmap)
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
        )


class _BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        pass


class _MMapBinReader(_BinReader):
    """A _BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_buffer_mmap = numpy.memmap(bin_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)  # type: ignore

    def read(self, dtype: type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        return numpy.frombuffer(
            self._bin_buffer, dtype=dtype, count=count, offset=offset
        )

    def __del__(self) -> None:
        """Clean up the object."""
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()  # type: ignore
        del self._bin_buffer_mmap


class _FileBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using a file pointer

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_path = bin_path

    def read(self, dtype: type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        sequence = numpy.empty(count, dtype=dtype)
        with open(self._bin_path, mode="rb", buffering=0) as bin_buffer_file:
            bin_buffer_file.seek(offset)
            bin_buffer_file.readinto(sequence)  # type: ignore
        return sequence


class _ListTokensBinReader(_BinReader):
    def __init__(self, tokens: list[int]) -> None:
        self._tokens = tokens

    def read(self, dtype: type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        offset = int(offset / DType.size(dtype))
        return numpy.array(self._tokens[offset : offset + count], dtype=dtype)


#
# class _S3BinReader(_BinReader):
#     """A _BinReader that reads from the data (.bin) file from S3
#
#     Args:
#         bin_path (str): bin_path (str): The path to the data (.bin) file.
#
#         bin_chunk_nbytes (int, optional): If not None, then maintain an in-memory cache to speed up calls to the `read` method. Furthermore, on a cache miss, download this number of bytes to refresh the cache. Otherwise (None), do not maintain an in-memory cache. A class that inherits from _BinReader may not implement caching in which case it should assert that `bin_chunk_nbytes` is None at initialization.
#     """
#
#     def __init__(self, bin_path: str, bin_chunk_nbytes: int) -> None:
#         assert bin_chunk_nbytes > 0
#         self._client = boto3.client("s3")
#         self._s3_bucket, self._s3_key = parse_s3_path(bin_path)
#         self._cache = None
#         self._cache_bytes_start = None
#         self._cache_bytes_end = None
#         self._cache_nbytes = bin_chunk_nbytes
#
#     def _extract_from_cache(self, offset: int, size: int) -> bytes:
#         """Extract `size` bytes starting at `offset` bytes into the cache"""
#         start = offset - self._cache_bytes_start
#         assert start >= 0
#         end = start + size
#         assert end <= len(self._cache)
#         return self._cache[start:end]
#
#     def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
#         """Read bytes into a numpy array.
#
#         Let `size` be the `count` * `DType.size(dtype)`. If the requested span of bytes [`offset`,
#         `offset` + `size`) is covered by the in-memory cache maintained by this class, then this
#         function extracts the requested span from that cache and returns it. Otherwise, this
#         function first refreshes the cache and then extracts the requested span from the refreshed
#         cache and returns it.
#
#         The cache is refreshed based on `offset` and `size`. In particular, we divide all the bytes
#         in an S3 object into blocks, where each block contains `bin_chunk_nbytes` bytes. We assign
#         each block an index starting from 0. We take the block with index (`offset` //
#         `bin_chunk_nbytes`) to refresh the cache. If this new block still does not cover the
#         requested span, we extend it just enough to include `offset` + `size`.
#
#         Args:
#             dtype (Type[numpy.number]): Data-type of the returned array.
#
#             count (int): Number of items to read.
#
#             offset (int): Start reading from this offset (in bytes).
#
#         Returns:
#             numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
#         """
#         size = count * DType.size(dtype)
#         if (
#             self._cache is not None
#             and offset >= self._cache_bytes_start
#             and offset + size <= self._cache_bytes_end
#         ):
#             return numpy.frombuffer(self._extract_from_cache(offset, size), dtype=dtype)
#
#         bytes_start = (offset // self._cache_nbytes) * self._cache_nbytes
#         assert bytes_start >= 0
#         assert offset >= bytes_start
#         bytes_end = max(bytes_start + self._cache_nbytes, offset + size)
#         assert bytes_end >= 1
#         self._cache = self._client.get_object(
#             Bucket=self._s3_bucket,
#             Key=self._s3_key,
#             # Subtract 1, because the end of Range is inclusive.
#             Range=f"bytes={bytes_start}-{bytes_end - 1}",
#         )["Body"].read()
#         self._cache_bytes_start = bytes_start
#         self._cache_bytes_end = bytes_end
#         return numpy.frombuffer(self._extract_from_cache(offset, size), dtype=dtype)
#
#     def __del__(self) -> None:
#         """Clean up the object"""
#         self._client.close()


class IndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool): Whether the dataset is multimodal. Defaults to False.

        mmap (bool): Whether to mmap the .bin files. Defaults to True.

        s3_config (Optional[S3Config]): Supplied only for data stored on S3. IndexedDataset downloads the index (.idx) file to `s3_config.path_to_idx_cache` and streams data from the data (.bin) file in `s3_config.bin_chunk_nbytes` blocks. Note that `mmap` must be disabled for S3 data loading. Defaults to None.
    """

    def __init__(
        self,
        path_prefix: str,
        multimodal: bool = False,
        mmap: bool = True,
        s3_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.path_prefix: str
        self.mmap: bool
        assert s3_config is None, s3_config

        self.index: _IndexReader
        self.bin_reader: _BinReader

        self.initialize(path_prefix, mmap)

    def initialize(self, path_prefix: str, mmap: bool) -> None:
        """Initialize the dataset

        This method is called by IndexedDataset.__init__ during object creation and by
        IndexedDataset.__setstate__ during un-pickling

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix

            multimodal (bool): Whether the dataset is multimodal

            mmap (bool): Whether to mmap the .bin file

            s3_config (Optional[S3Config]): See IndexedDataset docstring for details.
        """
        idx_path = get_idx_path(path_prefix)
        bin_path = get_bin_path(path_prefix)
        assert os.path.exists(idx_path) and os.path.exists(bin_path), (
            f"One or both of the .idx and .bin files cannot be found at the path prefix {path_prefix}"
        )
        self.path_prefix = path_prefix
        self.mmap = mmap
        if mmap:
            self.bin_reader = _MMapBinReader(bin_path)
        else:
            self.bin_reader = _FileBinReader(bin_path)
        self.index = _IndexReader(idx_path)

    def __getstate__(self) -> tuple[str, bool]:
        """Get the state during pickling

        Returns:
            Tuple[str, bool, bool, Optional[S3Config]]: The state tuple
        """
        return self.path_prefix, self.mmap

    def __setstate__(self, state: tuple[str, bool]) -> None:
        """Set the state during un-pickling

        Args:
            state (Tuple[str, bool, bool, Optional[S3Config]]): The state tuple
        """
        path_prefix, mmap = state
        self.initialize(path_prefix, mmap)

    def __del__(self) -> None:
        """Clean up the object"""
        del self.bin_reader
        del self.index

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    @overload
    def __getitem__(self, idx: int | numpy.integer) -> numpy.ndarray: ...

    @overload
    def __getitem__(self, idx: slice) -> list[numpy.ndarray]: ...

    def __getitem__(
        self, idx: int | numpy.integer | slice
    ) -> numpy.ndarray | list[numpy.ndarray]:
        """Return from the dataset

        Args:
            idx (Union[int, numpy.integer, slice]): The index or index slice into the dataset

        Raises:
            ValueError: When the index slice is non-contiguous

            TypeError: When the index is of an unexpected type

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and modes at the index or index slice
        """
        assert self.bin_reader is not None
        assert self.index is not None
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length = self.index[idx]
            return self.bin_reader.read(
                dtype=self.index.dtype,
                count=int(sequence_length),
                offset=int(sequence_pointer),
            )
        elif isinstance(idx, slice):
            start, _stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_offsets = list(accumulate(sequence_lengths))
            return numpy.split(
                self.bin_reader.read(
                    dtype=self.index.dtype,
                    count=int(sum(sequence_lengths)),
                    offset=int(self.index.sequence_pointers[start]),
                ),
                sequence_offsets[:-1],
            )
        else:
            assert_never(idx)

    def get(
        self, idx: int, offset: int = 0, length: int | None = None
    ) -> numpy.ndarray:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.

        Args:
            idx (Union[int, numpy.integer]): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (int): The number of tokens to grab from the sequence

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and modes at the index
        """
        sequence_pointer, sequence_length = self.index[idx]
        if length is None:
            length = int(sequence_length) - offset
        ptr = int(sequence_pointer) + offset * DType.size(self.index.dtype)
        return self.bin_reader.read(dtype=self.index.dtype, count=length, offset=ptr)

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        """Get the sequence lengths

        Returns:
            numpy.ndarray: The sequence lengths
        """
        return self.index.sequence_lengths

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Return whether the IndexedDataset exists on disk at the prefix

        Args:
            path_prefix (str): The prefix to the index (.idx) and data (.bin) files

        Returns:
            bool: Whether the IndexedDataset exists on disk at the prefix
        """
        # if is_s3_path(path_prefix):
        #     s3_client = boto3.client("s3")
        #     return object_exists(
        #         s3_client, get_idx_path(path_prefix)
        #     ) and object_exists(s3_client, get_bin_path(path_prefix))
        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(
            get_bin_path(path_prefix)
        )


class IndexedDatasetBuilder(object):
    """Builder class for the IndexedDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (Type[numpy.number], optional): The dtype of the index file. Defaults to numpy.int32.

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(
        self,
        path_prefix: str,
        dtype: type[numpy.number] = numpy.int32,
        multimodal: bool = False,
    ) -> None:
        self.path_prefix = path_prefix
        self.data_file = open(get_bin_path(path_prefix), "wb")
        self.dtype = dtype

        self._seq_lengths = numpy.empty(0, dtype=numpy.int32)
        self._pending: list[int] = []

    @classmethod
    def from_chunk(cls, src_prefix: str, dst_prefix: str) -> "IndexedDatasetBuilder":
        index = _IndexReader(get_idx_path(src_prefix))
        builder = cls(dst_prefix, dtype=index.dtype)
        builder._seq_lengths = index.sequence_lengths.copy()
        with open(get_bin_path(src_prefix), "rb") as f:
            shutil.copyfileobj(f, builder.data_file)
        return builder

    def _flush_pending(self) -> None:
        if self._pending:
            self._seq_lengths = numpy.append(
                self._seq_lengths, numpy.array(self._pending, dtype=numpy.int32)
            )
            self._pending = []

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        """Add a single item to the dataset

        Args:
            tensor (torch.Tensor): The item to add to the data file

            mode (int, optional): The mode for the item. Defaults to 0.
        """
        np_array = numpy.array(tensor.numpy(), dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self._pending.append(np_array.size)

    def add_document(
        self,
        tensor: torch.Tensor | list[list[int]],
        lengths: list[int],
        modes: Optional[list[int]] = None,
    ) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor): The document to add

            lengths (List[int]): The lengths of each item in the document

            modes (Optional[List[int]], optional): The modes for each item in the document. Defaults to None.
        """
        np_array = numpy.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self._pending.extend(lengths)
        if len(self._pending) >= 1_000_000:
            self._flush_pending()

    def end_document(self) -> None:
        """Finalize the document, for use with IndexedDatasetBuilder.add_item"""
        pass

    def merge_from(self, path_prefix: str) -> None:
        index = _IndexReader(get_idx_path(path_prefix))
        assert index.dtype == self.dtype

        self._flush_pending()
        self._seq_lengths = numpy.append(self._seq_lengths, index.sequence_lengths)

        with open(get_bin_path(path_prefix), "rb") as f:
            shutil.copyfileobj(f, self.data_file)

    def add_index(self, path_prefix: str) -> None:
        self.merge_from(path_prefix)

    def finalize(self, idx_path: str | None = None) -> None:
        self._flush_pending()
        self.data_file.close()
        with _IndexWriter(
            idx_path or get_idx_path(self.path_prefix), self.dtype
        ) as writer:
            writer.write(self._seq_lengths)


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the index file
    """
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the data file
    """
    return path_prefix + ".bin"
