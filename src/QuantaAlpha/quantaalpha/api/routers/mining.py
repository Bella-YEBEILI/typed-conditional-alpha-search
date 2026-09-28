from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from quantaalpha.api.deps import get_task_manager
from quantaalpha.api.schemas.common import TaskStatus
from quantaalpha.api.schemas.mining import (
    DirectionsResponse,
    MiningStartRequest,
    MiningTaskInfo,
    MiningProgress,
)
from quantaalpha.api.services.mining_service import MiningService
from quantaalpha.api.services.task_manager import TaskManager

router = APIRouter(prefix="/api/mining", tags=["mining"])


def get_mining_service() -> MiningService:
    return MiningService()


@router.get("/directions/{mode}", response_model=DirectionsResponse)
def get_directions(mode: str, svc: MiningService = Depends(get_mining_service)):
    return svc.get_directions(mode)


@router.post("/start")
def start_mining(
    body: MiningStartRequest,
    task_mgr: TaskManager = Depends(get_task_manager),
    svc: MiningService = Depends(get_mining_service),
):
    # 根据模式和选中的方向 ID 生成因子库后缀，如 pv_13_15_17
    direction_ids = body.direction_ids or []
    if direction_ids:
        ids_str = "_".join(str(i) for i in sorted(direction_ids))
        library_suffix = f"{body.mode}_{ids_str}"
    else:
        library_suffix = f"{body.mode}_custom"

    entry = task_mgr.create_task("mining", mode=body.mode, library_suffix=library_suffix)
    entry.extra["mode"] = body.mode
    entry.extra["library_suffix"] = library_suffix
    entry.extra["direction_ids"] = direction_ids
    entry.extra["config"] = body.config.model_dump()

    task_mgr.start_task(
        entry.task_id,
        target=svc.run_mining_task,
        kwargs={
            "mode": body.mode,
            "directions": body.directions,
            "custom_direction": body.custom_direction,
            "config": body.config.model_dump(),
            "stop_event": entry.stop_event,
            "library_suffix": library_suffix,
        },
    )

    return {
        "task_id": entry.task_id,
        "status": "started",
        "library_suffix": library_suffix,
        "message": f"挖掘任务已启动: mode={body.mode}, 因子库={library_suffix}",
    }


@router.get("/status/{task_id}")
def get_status(task_id: str, task_mgr: TaskManager = Depends(get_task_manager)):
    entry = task_mgr.get_task(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Task not found")
    elapsed = time.time() - entry.started_at if entry.started_at else 0
    progress = MiningProgress(**(entry.progress or {}))
    progress.elapsed_seconds = elapsed
    return MiningTaskInfo(
        task_id=entry.task_id,
        status=entry.status.value,
        mode=entry.extra.get("mode", ""),
        started_at=str(entry.started_at) if entry.started_at else "",
        progress=progress,
        error=entry.error,
    )


@router.post("/stop/{task_id}")
def stop_mining(task_id: str, task_mgr: TaskManager = Depends(get_task_manager)):
    ok = task_mgr.stop_task(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": "stopped"}


@router.get("/tasks")
def list_tasks(task_mgr: TaskManager = Depends(get_task_manager)):
    entries = task_mgr.list_tasks("mining")
    return [
        {
            "task_id": e.task_id,
            "status": e.status.value,
            "mode": e.extra.get("mode", ""),
            "started_at": str(e.started_at) if e.started_at else "",
        }
        for e in entries
    ]
