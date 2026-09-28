from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class PagedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[Any]


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
