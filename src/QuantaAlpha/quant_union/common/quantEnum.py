from __future__ import annotations

from enum import Enum


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class DomainType(_StrEnum):
    pv = "pv"
    fundamental = "fundamental"
    hybrid = "hybrid"


class CategoryType(_StrEnum):
    unknown = "unknown"
