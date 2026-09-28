from __future__ import annotations

from functools import lru_cache

from quantaalpha.api.services.task_manager import TaskManager
from quantaalpha.api.services.library_service import LibraryService


@lru_cache
def get_task_manager() -> TaskManager:
    return TaskManager()


def get_library_service() -> LibraryService:
    return LibraryService()
