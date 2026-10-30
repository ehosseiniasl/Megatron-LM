import abc
import copy
from typing import Generic, Hashable, Iterable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class DuplicatedKeyError(Exception):
    pass


class SaferDict(abc.ABC, Generic[K, V]):
    def __init__(self, dict_: dict[K, V] | None = None, *, _copy: bool = True) -> None:
        self._store: dict[K, V] = copy.copy(dict_) if dict_ is not None else {}
        self._copy = _copy

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._store:
            raise DuplicatedKeyError(f"{key} already specified")
        self._store[key] = copy.deepcopy(value) if self._copy else value

    def __getitem__(self, key: K) -> V:
        value = self._store[key]
        return copy.deepcopy(value) if self._copy else value

    def __contains__(self, key: K) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> Iterable[K]:
        return self._store.keys()

    def values(self) -> Iterable[V]:
        return self._store.values()

    def items(self) -> Iterable[tuple[K, V]]:
        return self._store.items()

    def to_dict(self) -> dict[K, V]:
        return copy.copy(self._store)
