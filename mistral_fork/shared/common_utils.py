from typing import Optional

from pydantic import NonNegativeInt, model_validator

from mistral_fork.mistral_common.base import MistralBase

TAR_HEADER_SIZE = 512


class TarError(Exception):
    pass


class TarOffset(MistralBase):
    offset: NonNegativeInt
    size: NonNegativeInt

    def __hash__(self) -> int:
        return hash((self.offset, self.size))

    new_tar_format: bool = False
    tar_path: Optional[str] = None

    @property
    def is_single_tar_format(self) -> bool:
        return self.tar_path is not None

    @model_validator(mode="after")
    def check_values(self) -> "TarOffset":
        if self.new_tar_format and self.size < TAR_HEADER_SIZE:
            raise ValueError(f"size {self.size} is incompatible with tar header size {TAR_HEADER_SIZE}")
        return self

    def from_within(self, parent_offset: "TarOffset") -> "TarOffset":
        if not self.offset + self.size <= parent_offset.size:
            raise TarError(f"Trying to read past end of parent: {self.offset=} + {self.size=} > {parent_offset.size=}")
        assert self.tar_path == parent_offset.tar_path
        offset = self.offset + parent_offset.offset
        if parent_offset.new_tar_format:
            offset += TAR_HEADER_SIZE
        return TarOffset(
            offset=offset,
            size=self.size,
            new_tar_format=self.new_tar_format,
            tar_path=self.tar_path,
        )
