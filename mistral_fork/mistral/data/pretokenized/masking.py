import dataclasses
from itertools import groupby


@dataclasses.dataclass(frozen=True)
class DocumentMask:
    sizes: list[int]
    mask: list[bool]

    def check(self) -> None:
        assert len(self.sizes) > 0, "mask must have at least one segment"
        assert all(size >= 1 for size in self.sizes), self
        for i in range(1, len(self.mask)):
            assert self.mask[i] != self.mask[i - 1], (
                f"adjacent segments must differ at index {i}: {self}"
            )

    def to_mask_index_entry(self) -> tuple[list[int], list[int]]:
        """Convert to MaskIndexEntry format: (sizes, masks as 0/1)."""
        return (self.sizes, [int(m) for m in self.mask])

    @classmethod
    def from_per_token_mask(cls, per_token_mask: list[bool]) -> "DocumentMask":
        assert len(per_token_mask) > 0, "per_token_mask must not be empty"
        sizes: list[int] = []
        mask: list[bool] = []
        for value, group in groupby(per_token_mask):
            sizes.append(sum(1 for _ in group))
            mask.append(value)
        return cls(sizes=sizes, mask=mask)
