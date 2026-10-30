import os
import warnings
from functools import lru_cache
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, model_validator


@lru_cache(maxsize=None)
def _warn_deprecated_field(class_name: str, field_name: str) -> None:
    warnings.warn(
        f"{field_name} is deprecated; it will be ignored.",
        FutureWarning,
        stacklevel=3,
    )


class MistralBase(BaseModel):
    _DEPRECATED_FIELDS: ClassVar[tuple[str, ...]] = ()

    model_config = ConfigDict(
        extra=("allow" if os.getenv("MISTRAL_BASE_EXTRA_ALLOWED") else "forbid"),
        validate_default=True,
        use_enum_values=True,
    )

    @model_validator(mode="before")
    def drop_deprecated_fields(cls, values: dict) -> dict:
        if not isinstance(values, dict):
            return values
        if any(field in values for field in getattr(cls, "_DEPRECATED_FIELDS", ())):
            values = values.copy()
        else:
            return values
        for field in getattr(cls, "_DEPRECATED_FIELDS", ()):
            try:
                values.pop(field)
                _warn_deprecated_field(cls.__name__, field)
            except KeyError:
                continue
        return values
