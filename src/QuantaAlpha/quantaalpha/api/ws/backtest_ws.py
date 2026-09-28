from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from quantaalpha.api.deps import get_task_manager

router = APIRouter()


@router.websocket("/ws/backtest/{task_id}")
async def backtest_websocket(websocket: WebSocket, task_id: str):
    task_mgr = get_task_manager()
    entry = task_mgr.get_task(task_id)
    if not entry:
        await websocket.close(code=4004, reason="Task not found")
        return

    await websocket.accept()
    client_queue: asyncio.Queue = asyncio.Queue()
    entry.ws_clients.append(client_queue)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(client_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            if msg is None:
                await websocket.send_json({
                    "type": "completed" if entry.status.value == "completed" else "error",
                    "data": {"error": entry.error} if entry.error else {},
                })
                break

            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        if client_queue in entry.ws_clients:
            entry.ws_clients.remove(client_queue)
