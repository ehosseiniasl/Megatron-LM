from typing import ClassVar, Protocol


class Args(Protocol):
    __dataclass_fields__: ClassVar[dict]
