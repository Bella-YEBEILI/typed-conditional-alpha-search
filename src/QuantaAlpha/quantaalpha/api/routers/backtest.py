from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from quantaalpha.api.deps import get_task_manager
from quantaalpha.api.schemas.backtest import (
    BacktestConfigResponse,
    BacktestRunRequest,
    BacktestResults,
    BacktestTaskStatus,
)
from quantaalpha.api.services.backtest_service import BacktestService
from quantaalpha.api.services.task_manager import TaskManager

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


def get_backtest_service() -> BacktestService:
    return BacktestService()


@router.get("/config", response_model=BacktestConfigResponse)
def get_config(svc: BacktestService = Depends(get_backtest_service)):
    return svc.get_config()


@router.post("/run")
def run_backtest(
    body: BacktestRunRequest,
    task_mgr: TaskManager = Depends(get_task_manager),
    svc: BacktestService = Depends(get_backtest_service),
):
    entry = task_mgr.create_task("backtest")

    task_mgr.start_task(
        entry.task_id,
        target=svc.run_backtest_task,
        kwargs={
            "factors": [f.model_dump() for f in body.factors],
            "config": body.config.model_dump(),
            "task_id": entry.task_id,
        },
    )

    return {
        "task_id": entry.task_id,
        "status": "started",
        "factor_count": len(body.factors),
    }


@router.get("/status/{task_id}", response_model=BacktestTaskStatus)
def get_status(task_id: str, task_mgr: TaskManager = Depends(get_task_manager)):
    entry = task_mgr.get_task(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Task not found")
    return BacktestTaskStatus(
        task_id=entry.task_id,
        status=entry.status.value,
        progress=entry.progress or {},
    )


@router.get("/results/{task_id}", response_model=BacktestResults)
def get_results(task_id: str, task_mgr: TaskManager = Depends(get_task_manager)):
    entry = task_mgr.get_task(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Task not found")
    factors = entry.result if isinstance(entry.result, list) else []
    return BacktestResults(task_id=task_id, factors=factors)


@router.get("/plot/{task_id}/{factor_name}")
def get_plot(task_id: str, factor_name: str):
    from quantaalpha.runtime import results_root
    plot_dir = results_root() / f"backtest_{task_id}" / factor_name
    for ext in ("_diagnostics.png", "_quality.png", ".png"):
        candidate = plot_dir / f"{factor_name}{ext}"
        if candidate.exists():
            from fastapi.responses import FileResponse
            return FileResponse(str(candidate), media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found")
