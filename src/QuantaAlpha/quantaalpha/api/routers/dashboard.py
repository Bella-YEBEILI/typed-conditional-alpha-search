from __future__ import annotations

from fastapi import APIRouter, Depends

from quantaalpha.api.deps import get_library_service, get_task_manager
from quantaalpha.api.schemas.dashboard import DashboardSummary
from quantaalpha.api.services.library_service import LibraryService
from quantaalpha.api.services.task_manager import TaskManager

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def get_summary(
    lib_svc: LibraryService = Depends(get_library_service),
    task_mgr: TaskManager = Depends(get_task_manager),
):
    summary = lib_svc.get_summary()
    summary["active_mining_tasks"] = task_mgr.active_count("mining")
    return summary
