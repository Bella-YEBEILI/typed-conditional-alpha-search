from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from quantaalpha.api.deps import get_task_manager
from quantaalpha.api.schemas.analysis import (
    AnalysisRunRequest,
    AnalysisResults,
    AnalysisTaskStatus,
)
from quantaalpha.api.services.analysis_service import AnalysisService
from quantaalpha.api.services.task_manager import TaskManager

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def get_analysis_service() -> AnalysisService:
    return AnalysisService()


@router.post("/run")
def run_analysis(
    body: AnalysisRunRequest,
    task_mgr: TaskManager = Depends(get_task_manager),
    svc: AnalysisService = Depends(get_analysis_service),
):
    entry = task_mgr.create_task("analysis")

    task_mgr.start_task(
        entry.task_id,
        target=svc.run_analysis_task,
        kwargs={
            "library_suffix": body.library_suffix,
            "quality_filter": body.quality_filter,
            "factor_ids": body.factor_ids,
            "task_id": entry.task_id,
        },
    )

    return {
        "task_id": entry.task_id,
        "status": "started",
    }


@router.get("/status/{task_id}", response_model=AnalysisTaskStatus)
def get_status(task_id: str, task_mgr: TaskManager = Depends(get_task_manager)):
    entry = task_mgr.get_task(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Task not found")
    return AnalysisTaskStatus(
        task_id=entry.task_id,
        status=entry.status.value,
        progress=entry.progress or {},
    )


@router.get("/results/{task_id}", response_model=AnalysisResults)
def get_results(task_id: str, task_mgr: TaskManager = Depends(get_task_manager)):
    entry = task_mgr.get_task(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Task not found")
    factors = entry.result if isinstance(entry.result, list) else []
    return AnalysisResults(task_id=task_id, factors=factors)


@router.get("/plot/{task_id}/{factor_id}")
def get_plot(task_id: str, factor_id: str):
    from quantaalpha.runtime import results_root
    plot_dir = results_root() / f"analysis_{task_id}" / factor_id
    for ext in ("_diagnostics.png", "_quality.png", ".png"):
        candidate = plot_dir / f"{factor_id}{ext}"
        if candidate.exists():
            from fastapi.responses import FileResponse
            return FileResponse(str(candidate), media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found")
