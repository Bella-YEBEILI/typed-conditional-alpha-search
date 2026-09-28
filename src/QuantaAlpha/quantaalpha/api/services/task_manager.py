from __future__ import annotations

import asyncio
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from quantaalpha.api.schemas.common import TaskStatus

# 日志文件存储目录: data/logs/
_LOGS_DIR = Path(__file__).resolve().parents[3] / "data" / "logs"


@dataclass
class TaskEntry:
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    started_at: float = 0.0
    finished_at: float = 0.0
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    log_queue: asyncio.Queue | None = None
    ws_clients: list[asyncio.Queue] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskEntry] = {}
        self._lock = threading.Lock()

    def create_task(self, task_type: str, **extra: Any) -> TaskEntry:
        task_id = uuid.uuid4().hex[:12]
        entry = TaskEntry(
            task_id=task_id,
            task_type=task_type,
            log_queue=asyncio.Queue(maxsize=5000),
            extra=extra,
        )
        with self._lock:
            self._tasks[task_id] = entry
        return entry

    def get_task(self, task_id: str) -> TaskEntry | None:
        return self._tasks.get(task_id)

    def list_tasks(self, task_type: str | None = None) -> list[TaskEntry]:
        with self._lock:
            entries = list(self._tasks.values())
        if task_type:
            entries = [e for e in entries if e.task_type == task_type]
        return entries

    def start_task(
        self,
        task_id: str,
        target: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> None:
        entry = self.get_task(task_id)
        if not entry:
            raise ValueError(f"Task {task_id} not found")
        entry.status = TaskStatus.RUNNING
        entry.started_at = time.time()
        entry.thread = threading.Thread(
            target=self._run_wrapper,
            args=(entry, target, args, kwargs or {}),
            daemon=True,
        )
        entry.thread.start()

    def stop_task(self, task_id: str) -> bool:
        entry = self.get_task(task_id)
        if not entry:
            return False
        entry.stop_event.set()
        entry.status = TaskStatus.STOPPED
        entry.finished_at = time.time()
        return True

    def active_count(self, task_type: str | None = None) -> int:
        tasks = self.list_tasks(task_type)
        return sum(1 for t in tasks if t.status == TaskStatus.RUNNING)

    def _run_wrapper(
        self,
        entry: TaskEntry,
        target: Callable,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """
        运行任务，捕获所有输出（loguru + print）并:
        1. 推送到 WebSocket 客户端（前端实时显示，和终端看到的一模一样）
        2. 写入 data/logs/{task_type}/{timestamp}_{task_id}.log 文件
        3. 终端不输出（不刷屏）
        """
        # 准备日志文件
        log_dir = _LOGS_DIR / entry.task_type
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir / f"{timestamp}_{entry.task_id}.log"
        log_file = open(log_file_path, "w", encoding="utf-8")
        log_file.write(
            f"=== Task: {entry.task_type} | ID: {entry.task_id} | "
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"Extra: {entry.extra}\n\n"
        )
        log_file.flush()

        current_thread_id = threading.current_thread().ident

        # ===== 1. 给 loguru 添加自定义 sink（捕获 mining 的全部日志输出）=====
        loguru_sink_id = None
        removed_handler_ids = []
        try:
            from loguru import logger as loguru_logger

            # 移除所有现有 handler（终端不输出）
            for hid in list(loguru_logger._core.handlers.keys()):
                removed_handler_ids.append(hid)
            for hid in removed_handler_ids:
                loguru_logger.remove(hid)

            # 添加我们的 sink：推送到 WS + 写入文件
            def _task_sink(message):
                # loguru message 是 str，末尾带换行
                text = str(message).rstrip("\n")
                if not text:
                    return
                _broadcast_to_ws(entry, text)
                try:
                    log_file.write(text + "\n")
                    log_file.flush()
                except Exception:
                    pass

            loguru_sink_id = loguru_logger.add(
                _task_sink,
                format="{time:HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                colorize=False,
            )
        except ImportError:
            pass

        # ===== 2. 拦截 print() 输出（stdout）=====
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _ThreadAwareStream(old_stdout, entry, log_file, current_thread_id)
        sys.stderr = _ThreadAwareStream(old_stderr, entry, log_file, current_thread_id)

        try:
            entry.result = target(*args, **kwargs)
            if entry.status == TaskStatus.RUNNING:
                entry.status = TaskStatus.COMPLETED
        except Exception as exc:
            entry.status = TaskStatus.FAILED
            entry.error = str(exc)
            error_msg = f"[FATAL] Task failed: {exc}"
            _broadcast_to_ws(entry, error_msg)
            try:
                log_file.write(error_msg + "\n")
            except Exception:
                pass
        finally:
            # 恢复 stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # 恢复 loguru（移除我们的 sink，恢复原来的终端 sink）
            try:
                from loguru import logger as loguru_logger
                if loguru_sink_id is not None:
                    loguru_logger.remove(loguru_sink_id)
                # 恢复默认 stderr sink
                loguru_logger.add(sys.stderr, level="DEBUG")
            except Exception:
                pass

            # 写结束标记并关闭文件
            try:
                log_file.write(
                    f"\n=== Task finished: status={entry.status.value} | "
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
                log_file.close()
            except Exception:
                pass

            entry.finished_at = time.time()

            # 通知所有 ws 客户端任务结束
            for q in list(entry.ws_clients):
                try:
                    q.put_nowait(None)
                except (asyncio.QueueFull, RuntimeError):
                    pass
            if entry.log_queue:
                try:
                    entry.log_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass


def _broadcast_to_ws(entry: TaskEntry, text: str) -> None:
    """推送日志消息到所有 WebSocket 客户端"""
    if not text.strip():
        return
    msg = {
        "type": "log",
        "message": text,
        "timestamp": time.time(),
    }
    for q in list(entry.ws_clients):
        try:
            q.put_nowait(msg)
        except (asyncio.QueueFull, RuntimeError):
            pass


class _ThreadAwareStream:
    """
    替换 sys.stdout/stderr，按线程区分:
    - 任务线程的 print() → WS + 日志文件（不输出到终端）
    - 其他线程（uvicorn等）→ 正常输出到终端
    """

    def __init__(self, original, entry: TaskEntry, log_file, target_thread_id: int):
        self._original = original
        self._entry = entry
        self._log_file = log_file
        self._target_thread_id = target_thread_id
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        if threading.current_thread().ident != self._target_thread_id:
            return self._original.write(text)

        # 任务线程：按行推送
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                _broadcast_to_ws(self._entry, line)
            try:
                self._log_file.write(line + "\n")
                self._log_file.flush()
            except Exception:
                pass
        return len(text)

    def flush(self):
        if threading.current_thread().ident != self._target_thread_id:
            self._original.flush()
            return
        if self._buffer.strip():
            _broadcast_to_ws(self._entry, self._buffer)
        if self._buffer:
            try:
                self._log_file.write(self._buffer)
                self._log_file.flush()
            except Exception:
                pass
            self._buffer = ""

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._original, name)
