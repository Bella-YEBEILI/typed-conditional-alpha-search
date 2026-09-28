"""
QuantaAlpha Backend API
FastAPI-based REST + WebSocket API for factor mining and backtesting.

Integrates with the core QuantaAlpha CLI to launch experiments
and reads factor library JSON for the factor browsing API.
"""

import asyncio
import glob
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file: frontend-v2/backend/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Ensure import quantaalpha is available (when backend is started from frontend-v2 directory, repo root is not in sys.path)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DOTENV_PATH = PROJECT_ROOT / ".env"
QUALITY_LIBRARY_NAMES = (
    "high_quality_factors_library.json",
    "medium_quality_factors_library.json",
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="QuantaAlpha API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================== Pydantic Models ==========================


class MiningStartRequest(BaseModel):
    """Request to start a factor mining experiment."""
    direction: str = Field(..., description="Research direction, e.g. '价量因子挖掘'")
    numDirections: Optional[int] = Field(2, description="Parallel exploration directions")
    maxRounds: Optional[int] = Field(3, description="Evolution rounds")
    maxLoops: Optional[int] = Field(2, description="Iterations per direction")
    factorsPerHypothesis: Optional[int] = Field(3, description="Factors per hypothesis")
    librarySuffix: Optional[str] = Field(None, description="Factor library file suffix")
    qualityGateEnabled: Optional[bool] = Field(None, description="Enable quality gate checks")
    parallelEnabled: Optional[bool] = Field(None, description="Enable parallel execution within evolution phases")


class BacktestStartRequest(BaseModel):
    """Request to start an independent backtest."""
    factorJson: str = Field(..., description="Path to factor library JSON")
    factorSource: str = Field("custom", description="custom | combined")
    configPath: Optional[str] = Field(None, description="Path to backtest config")


class SystemConfigUpdate(BaseModel):
    """Partial update to system configuration (.env)."""
    QUANTAALPHA_DATA_ROOT: Optional[str] = None
    DATA_RESULTS_DIR: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    CHAT_MODEL: Optional[str] = None
    REASONING_MODEL: Optional[str] = None


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None


# ========================== In-Memory State ==========================

tasks: Dict[str, Dict[str, Any]] = {}
ws_connections: Dict[str, List[WebSocket]] = {}  # task_id -> list of WS


# ========================== Utility Helpers ==========================

def _gen_id() -> str:
    return str(uuid.uuid4())[:8]


def _now() -> str:
    return datetime.now().isoformat()


def _load_dotenv_dict() -> Dict[str, str]:
    """Parse the .env file into a dict (simple key=value, stripping quotes and inline comments)."""
    env: Dict[str, str] = {}
    if DOTENV_PATH.exists():
        for line in DOTENV_PATH.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            val = val.strip()
            if len(val) >= 2 and val[0] in ("'", '"'):
                quote = val[0]
                end = val.find(quote, 1)
                if end != -1:
                    val = val[1:end]
            else:
                idx = val.find(" #")
                if idx != -1:
                    val = val[:idx].strip()
            env[key.strip()] = val
    return env


def _find_factor_jsons() -> List[str]:
    """Find all factor library JSON files in data/factorlib/."""
    factorlib_dir = PROJECT_ROOT / "data" / "factorlib"
    pattern = str(factorlib_dir / "all_factors_library*.json")
    results = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    quality_results = []
    for name in QUALITY_LIBRARY_NAMES:
        quality_path = factorlib_dir / name
        if quality_path.exists():
            quality_results.append(str(quality_path))
    results = quality_results + results

    old_pattern = str(PROJECT_ROOT / "all_factors_library*.json")
    old_results = sorted(glob.glob(old_pattern), key=os.path.getmtime, reverse=True)

    seen = set(results)
    for r in old_results:
        if r not in seen:
            results.append(r)
    return results


def _source_factor_jsons() -> List[str]:
    return [p for p in _find_factor_jsons() if Path(p).name not in QUALITY_LIBRARY_NAMES]


def _load_factor_library(path: str) -> Dict[str, Any]:
    """Load and parse a factor library JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _classify_quality(factor_info: Dict[str, Any], backtest_results: Dict[str, Any] = None) -> str:
    """Classify factor quality.

    High: test_passed AND train_passed both True
    Medium: only one of test_passed / train_passed True
    Low: neither passed
    """
    train_flags = factor_info.get("train_check_flags") or {}
    test_flags = factor_info.get("test_check_flags") or {}
    trp = bool(train_flags.get("check_passed", False))
    tp = bool(test_flags.get("check_passed", False))

    if tp and trp:
        return "high"
    if tp or trp:
        return "medium"
    return "low"


def _quality_counts_for_library(path: str) -> Dict[str, int]:
    """Count factors by quality for one factor library."""
    try:
        raw = _load_factor_library(path)
    except Exception:
        return {"high": 0, "medium": 0, "low": 0}

    counts = {"high": 0, "medium": 0, "low": 0}
    factors = raw.get("factors", {})
    if not isinstance(factors, dict):
        return counts

    for factor_info in factors.values():
        if not isinstance(factor_info, dict):
            continue
        bt = factor_info.get("test_backtest_metrics", factor_info.get("backtest_results", {}))
        if not isinstance(bt, dict):
            bt = {}
        q = _classify_quality(factor_info, bt)
        counts[q] = counts.get(q, 0) + 1
    return counts


def _ensure_quality_bucket_libraries(factorlib_dir: Optional[Path] = None) -> List[str]:
    """Materialize backend high-only and medium-only factor libraries."""
    output_dir = factorlib_dir or (PROJECT_ROOT / "data" / "factorlib")
    output_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, Dict[str, Any]] = {"high": {}, "medium": {}}
    sources: Dict[str, Dict[str, str]] = {"high": {}, "medium": {}}

    for lib_path in _source_factor_jsons():
        try:
            raw = _load_factor_library(lib_path)
        except Exception:
            continue
        factors = raw.get("factors", {})
        if not isinstance(factors, dict):
            continue
        for factor_id, factor_info in factors.items():
            if not isinstance(factor_info, dict):
                continue
            bt = factor_info.get("test_backtest_metrics", factor_info.get("backtest_results", {}))
            if not isinstance(bt, dict):
                bt = {}
            quality = _classify_quality(factor_info, bt)
            if quality not in buckets or factor_id in buckets[quality]:
                continue
            copied = dict(factor_info)
            copied.setdefault("factor_id", factor_id)
            copied["quality_bucket_source_library"] = Path(lib_path).name
            buckets[quality][factor_id] = copied
            sources[quality][factor_id] = Path(lib_path).name

    written: List[str] = []
    now = _now()
    for quality, filename in (
        ("high", "high_quality_factors_library.json"),
        ("medium", "medium_quality_factors_library.json"),
    ):
        out_path = output_dir / filename
        payload = {
            "metadata": {
                "quality": quality,
                "total_factors": len(buckets[quality]),
                "last_updated": now,
                "source": "auto-generated from backend factor libraries",
                "source_libraries": sorted(set(sources[quality].values())),
            },
            "factors": buckets[quality],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written.append(str(out_path))
    return written


def _displayable_factor_jsons() -> List[str]:
    """Show only libraries that contain at least one medium-or-better factor."""
    _ensure_quality_bucket_libraries()
    displayable: List[str] = []
    for path in _find_factor_jsons():
        counts = _quality_counts_for_library(path)
        if counts.get("high", 0) + counts.get("medium", 0) > 0:
            displayable.append(path)
    return displayable


def _positive_int(value: Optional[int], default: int) -> int:
    try:
        return max(1, int(value if value is not None else default))
    except (TypeError, ValueError):
        return max(1, int(default))


def _effective_mining_settings(req: MiningStartRequest) -> Dict[str, Any]:
    """Normalize frontend mining settings before writing runtime config."""
    return {
        "numDirections": _positive_int(req.numDirections, 2),
        "maxRounds": _positive_int(req.maxRounds, 3),
        "maxLoops": _positive_int(req.maxLoops, 2),
        "factorsPerHypothesis": _positive_int(req.factorsPerHypothesis, 3),
        "librarySuffix": req.librarySuffix,
        "parallelEnabled": req.parallelEnabled,
        "qualityGateEnabled": req.qualityGateEnabled,
    }


def _apply_mining_overrides(run_cfg: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply effective frontend mining settings to the YAML config used by the subprocess."""
    run_cfg.setdefault("planning", {})["num_directions"] = settings["numDirections"]
    run_cfg.setdefault("evolution", {})["max_rounds"] = settings["maxRounds"]
    run_cfg.setdefault("execution", {})["max_loops"] = settings["maxLoops"]
    run_cfg.setdefault("factor", {})["factors_per_hypothesis"] = settings["factorsPerHypothesis"]

    if settings.get("parallelEnabled") is not None:
        run_cfg.setdefault("evolution", {})["parallel_enabled"] = settings["parallelEnabled"]
        run_cfg.setdefault("execution", {})["parallel_execution"] = settings["parallelEnabled"]

    if settings.get("qualityGateEnabled") is not None:
        qg = run_cfg.setdefault("quality_gate", {})
        if settings["qualityGateEnabled"]:
            qg.setdefault("complexity_enabled", True)
            qg.setdefault("redundancy_enabled", True)
            qg.setdefault("consistency_enabled", False)
        else:
            qg["consistency_enabled"] = False
            qg["complexity_enabled"] = False
            qg["redundancy_enabled"] = False

    return run_cfg


async def _broadcast(task_id: str, message: Dict[str, Any]):
    """Send a JSON message to all WebSocket clients for a task."""
    if task_id not in ws_connections:
        return
    dead: List[WebSocket] = []
    for ws in ws_connections[task_id]:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections[task_id].remove(ws)


# ========================== Mining Process ==========================

async def _run_mining(task_id: str, req: MiningStartRequest):
    """
    Launch the actual QuantaAlpha mining experiment as a subprocess
    and stream its output over WebSocket.
    """
    task = tasks[task_id]
    try:
        # Build the command
        env = os.environ.copy()
        # Load .env into env
        dotenv = _load_dotenv_dict()
        env.update(dotenv)

        # Use experiment_id as suffix to guarantee isolation
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        env["EXPERIMENT_ID"] = experiment_id
        
        # Enforce unique library suffix if not provided
        if not req.librarySuffix:
            req.librarySuffix = experiment_id
            # Update task config so frontend knows the suffix
            task["config"]["librarySuffix"] = req.librarySuffix

        mining_settings = _effective_mining_settings(req)
        task["config"].update(mining_settings)

        env["FACTOR_LIBRARY_SUFFIX"] = req.librarySuffix
        env["FACTOR_FACTORS_PER_HYPOTHESIS"] = str(mining_settings["factorsPerHypothesis"])

        results_base = dotenv.get("DATA_RESULTS_DIR", str(PROJECT_ROOT / "data" / "results"))
        env["WORKSPACE_PATH"] = f"{results_base}/workspace_{experiment_id}"
        env["PICKLE_CACHE_FOLDER_PATH_STR"] = f"{results_base}/pickle_cache_{experiment_id}"

        os.makedirs(env["WORKSPACE_PATH"], exist_ok=True)
        os.makedirs(env["PICKLE_CACHE_FOLDER_PATH_STR"], exist_ok=True)

        # Build a temporary config with frontend parameter overrides
        base_config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
        config_path_to_use = str(base_config_path)

        try:
            with open(base_config_path, "r", encoding="utf-8") as _f:
                run_cfg = yaml.safe_load(_f) or {}

            _apply_mining_overrides(run_cfg, mining_settings)

            # Write to a temporary file so the original is untouched
            tmp_dir = Path(env.get("WORKSPACE_PATH", "/tmp"))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_cfg = tmp_dir / "experiment_override.yaml"
            with open(tmp_cfg, "w", encoding="utf-8") as _f:
                yaml.safe_dump(run_cfg, _f, allow_unicode=True, default_flow_style=False)
            config_path_to_use = str(tmp_cfg)
        except Exception as cfg_err:
            # Fall back to original config if anything fails
            import traceback
            traceback.print_exc()

        # Build CLI args
        cmd = [
            sys.executable, "-m", "quantaalpha.cli", "mine",
            "--direction", req.direction,
            "--config_path", config_path_to_use,
        ]

        task["status"] = "running"
        task["progress"]["phase"] = "planning"
        task["progress"]["message"] = "正在启动实验..."
        task["updatedAt"] = _now()

        applied_log = {
            "id": _gen_id(),
            "timestamp": _now(),
            "level": "info",
            "message": (
                "Applied mining settings: "
                f"num_directions={mining_settings['numDirections']}, "
                f"max_rounds={mining_settings['maxRounds']}, "
                f"max_loops={mining_settings['maxLoops']}, "
                f"factors_per_hypothesis={mining_settings['factorsPerHypothesis']}, "
                f"library_suffix={mining_settings['librarySuffix']}, "
                f"parallel_enabled={mining_settings['parallelEnabled']}, "
                f"quality_gate_enabled={mining_settings['qualityGateEnabled']}, "
                f"config_path={config_path_to_use}"
            ),
        }
        task["logs"].append(applied_log)

        await _broadcast(task_id, {
            "type": "progress",
            "taskId": task_id,
            "data": task["progress"],
            "timestamp": _now(),
        })
        await _broadcast(task_id, {
            "type": "log",
            "taskId": task_id,
            "data": applied_log,
            "timestamp": _now(),
        })

        # Launch subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        task["pid"] = proc.pid

        # Stream stdout line by line
        line_count = 0
        current_phase = "planning"

        # Noisy patterns to suppress (shared with backtest)
        _MINING_NOISE = (
            "field data contains nan",
            "common_infra",
            "PyTorch models are skipped",
            "UserWarning: pkg_resources",
            "FutureWarning",
            "UserWarning",
            "Training until validation scores",
            "Did not meet early stopping",
        )

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            line_count += 1

            # Skip noisy warnings
            if any(p in line for p in _MINING_NOISE):
                continue

            # Detect phase from log messages
            new_phase = current_phase
            if "factor_propose" in line:
                new_phase = "evolving"
            elif "factor_backtest" in line or "backtest" in line.lower():
                new_phase = "backtesting"
            elif "feedback" in line:
                new_phase = "analyzing"
            elif "factor_calculate" in line:
                new_phase = "evolving"
            elif "规划" in line or "planning" in line.lower():
                new_phase = "planning"
            elif "进化完成" in line or "程序执行完成" in line:
                new_phase = "completed"

            if new_phase != current_phase:
                current_phase = new_phase
                task["progress"]["phase"] = current_phase
                task["progress"]["message"] = line[:200]
                task["progress"]["timestamp"] = _now()
                await _broadcast(task_id, {
                    "type": "progress",
                    "taskId": task_id,
                    "data": task["progress"],
                    "timestamp": _now(),
                })

            # Send log every line (throttle to avoid flooding)
            if line_count % 3 == 0 or "INFO" in line or "ERROR" in line or "WARNING" in line:
                level = "info"
                if "ERROR" in line or "Error" in line:
                    level = "error"
                elif "WARNING" in line or "Warning" in line:
                    level = "warning"
                elif "完成" in line or "success" in line.lower():
                    level = "success"

                log_entry = {
                    "id": _gen_id(),
                    "timestamp": _now(),
                    "level": level,
                    "message": line[:500],
                }
                task["logs"].append(log_entry)
                # Keep only last 500 logs in memory
                if len(task["logs"]) > 500:
                    task["logs"] = task["logs"][-500:]

                await _broadcast(task_id, {
                    "type": "log",
                    "taskId": task_id,
                    "data": log_entry,
                    "timestamp": _now(),
                })

            # Extract metrics from log lines like "RankIC=0.0016"
            if "RankIC=" in line:
                try:
                    rank_ic_str = line.split("RankIC=")[1].split(",")[0].split(")")[0]
                    task["metrics"]["rankIc"] = float(rank_ic_str)
                    await _broadcast(task_id, {
                        "type": "metrics",
                        "taskId": task_id,
                        "data": task["metrics"],
                        "timestamp": _now(),
                    })
                except Exception:
                    pass
            
            # Check for factor saving to update top factors list
            if "Saved factors to library" in line or "Saved" in line and "factors to" in line:
                _update_mining_metrics(task)
                if task.get("metrics"):
                     await _broadcast(task_id, {
                        "type": "result",
                        "taskId": task_id,
                        "data": {"status": task["status"], "metrics": task["metrics"]},
                        "timestamp": _now(),
                    })

        exit_code = await proc.wait()
        task["pid"] = None

        if exit_code == 0:
            task["status"] = "completed"
            task["progress"]["phase"] = "completed"
            task["progress"]["progress"] = 100
            task["progress"]["message"] = "实验完成"
        else:
            task["status"] = "failed"
            task["progress"]["message"] = f"实验失败 (exit code: {exit_code})"

        task["updatedAt"] = _now()

        # Load final factor count from the library JSON
        # Prefer the library file matching the librarySuffix for this experiment
        _update_mining_metrics(task)

        await _broadcast(task_id, {
            "type": "result",
            "taskId": task_id,
            "data": {"status": task["status"], "metrics": task["metrics"]},
            "timestamp": _now(),
        })

    except Exception as e:
        task["status"] = "failed"
        task["progress"]["message"] = f"Error: {str(e)}"
        task["updatedAt"] = _now()
        await _broadcast(task_id, {
            "type": "error",
            "taskId": task_id,
            "data": {"error": str(e)},
            "timestamp": _now(),
        })


# ========================== API Endpoints ==========================

@app.get("/")
async def root():
    # Serve frontend SPA if .build/index.html exists
    _idx = Path(__file__).resolve().parent.parent / ".build" / "index.html"
    if _idx.exists():
        from fastapi.responses import HTMLResponse as _HR
        return _HR(_idx.read_text(encoding="utf-8"))
    return {"message": "QuantaAlpha API", "version": "2.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": _now()}


# ---- Mining endpoints ----

@app.post("/api/v1/mining/start", response_model=ApiResponse)
async def start_mining(req: MiningStartRequest):
    """Start a new factor mining experiment."""
    task_id = _gen_id()
    task = {
        "taskId": task_id,
        "status": "running",
        "config": req.model_dump(),
        "progress": {
            "phase": "parsing",
            "currentRound": 0,
            "totalRounds": req.maxRounds or 3,
            "progress": 0,
            "message": "正在初始化实验...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {
            "ic": 0, "icir": 0, "rankIc": 0, "rankIcir": 0,
            "annualReturn": 0, "sharpeRatio": 0, "maxDrawdown": 0,
            "totalFactors": 0, "highQualityFactors": 0,
            "mediumQualityFactors": 0, "lowQualityFactors": 0,
        },
        "result": None,
        "pid": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task

    # Launch the mining process in background
    asyncio.create_task(_run_mining(task_id, req))

    return ApiResponse(
        success=True,
        data={"taskId": task_id, "task": task},
        message="实验已启动",
    )


@app.get("/api/v1/mining/{task_id}", response_model=ApiResponse)
async def get_mining_status(task_id: str):
    """Get task status."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data={"task": tasks[task_id]})


@app.delete("/api/v1/mining/{task_id}", response_model=ApiResponse)
async def cancel_mining(task_id: str):
    """Cancel a running mining task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.get("pid"):
        try:
            pid = task["pid"]
            # Try graceful termination first
            os.kill(pid, signal.SIGTERM)
            
            # Wait briefly for cleanup (0.5s)
            for _ in range(5):
                try:
                    os.kill(pid, 0) # Check if alive
                    await asyncio.sleep(0.1)
                except ProcessLookupError:
                    break
            
            # Force kill if still running
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
    task["status"] = "cancelled"
    task["updatedAt"] = _now()
    await _broadcast(task_id, {
        "type": "result",
        "taskId": task_id,
        "data": {"status": "cancelled"},
        "timestamp": _now(),
    })
    return ApiResponse(success=True, message="任务已取消")


@app.get("/api/v1/mining/tasks/list", response_model=ApiResponse)
async def list_tasks():
    """List all tasks."""
    task_list = sorted(tasks.values(), key=lambda t: t["createdAt"], reverse=True)
    return ApiResponse(success=True, data={"tasks": task_list})


# ---- Initial Directions endpoint ----

@app.get("/api/v1/directions/{domain}", response_model=ApiResponse)
async def get_initial_directions(domain: str):
    """Get available initial direction portfolios for a given domain (pv/minutes/joint)."""
    domain_map = {
        "pv": "daily",
        "minutes": "minutes",
        "joint": "joint",
    }
    base_name = domain_map.get(domain, domain)
    experiment_dir = PROJECT_ROOT / "experiment"
    candidates = [
        experiment_dir / f"original_direction_{base_name}.json",
        experiment_dir / f"original_direction_{domain}.json",
    ]

    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                portfolios = data.get("factor_portfolios", [])
                return ApiResponse(
                    success=True,
                    data={
                        "domain": domain,
                        "portfolios": portfolios,
                        "total": len(portfolios),
                    },
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    return ApiResponse(success=True, data={"domain": domain, "portfolios": [], "total": 0})


# ---- Factor library endpoints ----

@app.get("/api/v1/factors", response_model=ApiResponse)
async def get_factors(
    quality: Optional[str] = Query(None, description="Filter by quality: high/medium/low"),
    search: Optional[str] = Query(None, description="Search by factor name"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    library: Optional[str] = Query(None, description="Specific library file name"),
):
    """Get factors from the factor library JSON."""
    # Find the most recent factor library
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        # Fallback: check if file exists at project root (legacy location)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _displayable_factor_jsons()
        if not jsons:
            return ApiResponse(
                success=True,
                data={"factors": [], "total": 0, "limit": limit, "offset": offset,
                      "libraries": []},
            )
        lib_path = jsons[0]

    try:
        raw = _load_factor_library(lib_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read factor library: {e}")

    factors_dict = raw.get("factors", {})
    metadata = raw.get("metadata", {})

    # Convert dict to list with quality classification
    factors_list: List[Dict[str, Any]] = []
    for factor_id, factor_info in factors_dict.items():
        if not isinstance(factor_info, dict):
            continue
        # Read metrics from test_backtest_metrics (primary) or backtest_results (legacy)
        bt = factor_info.get("test_backtest_metrics", factor_info.get("backtest_results", {}))
        if not isinstance(bt, dict):
            bt = {}
        q = _classify_quality(factor_info, bt)

        # Helper to safely extract numeric values (handle NaN, None, non-numeric)
        def _safe_num(val, default=0):
            if val is None:
                return default
            try:
                v = float(val)
                import math
                return default if math.isnan(v) or math.isinf(v) else v
            except (TypeError, ValueError):
                return default

        # Extract IC metrics from test_backtest_metrics keys
        rank_ic = _safe_num(bt.get("rankic", bt.get("rank_ic", bt.get("Rank IC", bt.get("1day.excess_return_without_cost.rank_ic", 0)))))
        rank_icir = _safe_num(bt.get("rankicir", bt.get("rank_ic_ir", bt.get("Rank ICIR", bt.get("1day.excess_return_without_cost.rank_ic_ir", 0)))))
        ic = _safe_num(bt.get("ic", bt.get("IC", bt.get("1day.excess_return_without_cost.information_coefficient", rank_ic))))
        icir = _safe_num(bt.get("icir", bt.get("ICIR", bt.get("1day.excess_return_without_cost.information_coefficient_ir", rank_icir))))

        # Extract return/risk metrics
        annual_return = _safe_num(bt.get("long_ret", bt.get("long_netret", bt.get("1day.excess_return_with_cost.annualized_return", 0))))
        max_drawdown = _safe_num(bt.get("long_maxdd", bt.get("long_netmaxdd", bt.get("1day.excess_return_with_cost.max_drawdown", 0))))
        sharpe_ratio = _safe_num(bt.get("long_ir", bt.get("long_netir", bt.get("1day.excess_return_with_cost.information_ratio", 0))))

        # Round from metadata
        meta = factor_info.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}

        # Direction: derive short domain label from factor_level
        fl = factor_info.get("factor_level", "")
        if fl == "days":
            dir_label = "pv"
        elif fl == "minutes":
            dir_label = "minutes"
        elif fl == "joint" or fl == "cross_sectional":
            dir_label = "joint"
        else:
            dir_label = fl or ""

        factor_entry = {
            "factorId": factor_info.get("factor_id", factor_id),
            "factorName": factor_info.get("factor_name", "Unknown"),
            "factorExpression": factor_info.get("factor_expression", ""),
            "factorDescription": factor_info.get("factor_description", ""),
            "factorFormulation": factor_info.get("factor_formulation", ""),
            "quality": q,
            "backtestResults": bt,
            "ic": ic,
            "icir": icir,
            "rankIc": rank_ic,
            "rankIcir": rank_icir,
            "annualReturn": annual_return,
            "maxDrawdown": max_drawdown,
            "sharpeRatio": sharpe_ratio,
            "round": meta.get("round_number", meta.get("round", 0)),
            "direction": dir_label,
            "createdAt": factor_info.get("added_at", ""),
        }
        factors_list.append(factor_entry)

    # Apply filters
    if quality:
        factors_list = [f for f in factors_list if f["quality"] == quality]
    if search:
        search_lower = search.lower()
        factors_list = [
            f for f in factors_list
            if search_lower in f["factorName"].lower()
            or search_lower in f.get("factorDescription", "").lower()
            or search_lower in f.get("factorExpression", "").lower()
        ]

    _quality_rank = {"high": 0, "medium": 1, "low": 2}
    factors_list.sort(key=lambda f: (_quality_rank.get(f["quality"], 3), -abs(f.get("rankIc", 0))))

    total = len(factors_list)
    paginated = factors_list[offset: offset + limit]

    # Available library files
    all_libs = [Path(p).name for p in _displayable_factor_jsons()]

    return ApiResponse(
        success=True,
        data={
            "factors": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "metadata": metadata,
            "libraries": all_libs,
        },
    )


# ---- Factor cache endpoints ----
# IMPORTANT: These must be registered BEFORE /api/v1/factors/{factor_id}
# otherwise FastAPI matches "cache-status" as a factor_id parameter.

@app.get("/api/v1/factors/cache-status", response_model=ApiResponse)
async def get_cache_status(
    library: Optional[str] = Query(None, description="Factor library JSON filename"),
):
    """Check cache status of factors in the specified factor library."""
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _find_factor_jsons()
        if not jsons:
            return ApiResponse(success=True, data={
                "total": 0, "h5_cached": 0, "md5_cached": 0,
                "need_compute": 0, "factors": [],
            })
        lib_path = jsons[0]

    if not Path(lib_path).exists():
        raise HTTPException(status_code=404, detail=f"Factor library not found: {library}")

    # Import from core library
    from quantaalpha.factors.library import FactorLibraryManager
    result = FactorLibraryManager.check_cache_status(lib_path)
    return ApiResponse(success=True, data=result)


@app.post("/api/v1/factors/warm-cache", response_model=ApiResponse)
async def warm_cache(
    library: Optional[str] = Query(None, description="Factor library JSON filename"),
):
    """Batch sync from result.h5 to MD5 cache directory."""
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _find_factor_jsons()
        if not jsons:
            return ApiResponse(success=False, error="未找到因子库文件")
        lib_path = jsons[0]

    if not Path(lib_path).exists():
        raise HTTPException(status_code=404, detail=f"Factor library not found: {library}")

    from quantaalpha.factors.library import FactorLibraryManager
    result = FactorLibraryManager.warm_cache_from_json(lib_path)
    # Build a clear message
    parts = []
    if result['synced']:
        parts.append(f"新同步 {result['synced']} 个")
    if result.get('already_cached'):
        parts.append(f"已有缓存 {result['already_cached']} 个")
    if result.get('no_source'):
        parts.append(f"无H5源 {result['no_source']} 个(回测时从表达式计算)")
    if result['failed']:
        parts.append(f"失败 {result['failed']} 个")
    msg = "，".join(parts) if parts else "无需操作"
    return ApiResponse(
        success=True,
        data=result,
        message=msg,
    )


# ---- Factor library list endpoint (must be BEFORE {factor_id} route) ----

@app.get("/api/v1/factors/libraries", response_model=ApiResponse)
async def list_factor_libraries():
    """List factor libraries that have at least one medium-or-better factor."""
    libs = [Path(p).name for p in _displayable_factor_jsons()]
    return ApiResponse(success=True, data={"libraries": libs})


@app.get("/api/v1/factors/{factor_id}", response_model=ApiResponse)
async def get_factor_detail(factor_id: str):
    """Get full detail of a single factor."""
    jsons = _find_factor_jsons()
    for lib_path in jsons:
        try:
            raw = _load_factor_library(lib_path)
            factors = raw.get("factors", {})
            if factor_id in factors:
                info = factors[factor_id]
                return ApiResponse(success=True, data={"factor": info})
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="Factor not found")


# ---- Backtest endpoints ----

@app.post("/api/v1/backtest/start", response_model=ApiResponse)
async def start_backtest(req: BacktestStartRequest):
    """Start an independent backtest."""
    task_id = _gen_id()
    config_path = req.configPath or str(PROJECT_ROOT / "configs" / "backtest.yaml")

    task = {
        "taskId": task_id,
        "status": "running",
        "type": "backtest",
        "config": {**req.model_dump(), "configPath": config_path},
        "progress": {
            "phase": "backtesting",
            "currentRound": 0,
            "totalRounds": 1,
            "progress": 0,
            "message": "正在启动回测...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {},
        "result": None,
        "pid": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task

    # Launch backtest in background
    asyncio.create_task(_run_backtest(task_id, req, config_path))
    return ApiResponse(
        success=True,
        data={"taskId": task_id, "task": task},
        message="回测已启动",
    )


@app.get("/api/v1/backtest/{task_id}", response_model=ApiResponse)
async def get_backtest_status(task_id: str):
    """Get backtest task status and results."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data={"task": tasks[task_id]})


@app.delete("/api/v1/backtest/{task_id}", response_model=ApiResponse)
async def cancel_backtest(task_id: str):
    """Cancel a running backtest task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.get("pid"):
        try:
            os.kill(task["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    task["status"] = "cancelled"
    task["updatedAt"] = _now()
    await _broadcast(task_id, {
        "type": "result",
        "taskId": task_id,
        "data": {"status": "cancelled"},
        "timestamp": _now(),
    })
    return ApiResponse(success=True, message="回测已取消")


async def _run_backtest(task_id: str, req: BacktestStartRequest, config_path: str):
    """Run the independent backtest (V2) as a subprocess."""
    task = tasks[task_id]
    try:
        env = os.environ.copy()
        dotenv = _load_dotenv_dict()
        env.update(dotenv)

        # --- Resolve factor JSON path ---
        # Frontend sends just the filename (e.g. "all_factors_library_test3hjback.json")
        # We need to resolve it to the full path under data/factorlib/
        factor_json_input = req.factorJson
        factor_json_path = Path(factor_json_input)
        if not factor_json_path.is_absolute():
            # Check data/factorlib/ first
            candidate = PROJECT_ROOT / "data" / "factorlib" / factor_json_input
            if candidate.exists():
                factor_json_path = candidate
            else:
                # Try as relative to project root
                candidate2 = PROJECT_ROOT / factor_json_input
                if candidate2.exists():
                    factor_json_path = candidate2
                else:
                    factor_json_path = candidate  # will fail with a clear error message
        factor_json_str = str(factor_json_path)

        # --- Find the correct Python executable ---
        # Prefer the conda env that has qlib installed
        conda_env = dotenv.get("CONDA_ENV_NAME", "quantaalpha")
        python_bin = sys.executable  # fallback

        # Dynamically detect conda base path (portable, no hardcoded paths)
        conda_prefixes = [os.path.expanduser(f"~/.conda/envs/{conda_env}")]
        try:
            import subprocess as _sp
            conda_base = _sp.check_output(
                ["conda", "info", "--base"], text=True, timeout=5
            ).strip()
            conda_prefixes.insert(0, os.path.join(conda_base, "envs", conda_env))
        except Exception:
            pass
        # Also check CONDA_PREFIX if we're already in the right env
        if os.environ.get("CONDA_PREFIX"):
            conda_prefixes.insert(0, os.environ["CONDA_PREFIX"])

        for prefix in conda_prefixes:
            candidate_bin = Path(prefix) / "bin" / "python"
            if candidate_bin.exists():
                python_bin = str(candidate_bin)
                break

        # Build CLI command
        cmd = [
            python_bin, "-m", "quantaalpha.backtest.run_backtest",
            "-c", config_path,
            "--factor-source", req.factorSource,
            "--factor-json", factor_json_str,
            "--skip-uncached",
            "-v",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        task["pid"] = proc.pid

        # Noisy warnings from Qlib / dependencies that can be safely suppressed
        _NOISY_PATTERNS = (
            "field data contains nan",
            "common_infra",
            "PyTorch models are skipped",
            "UserWarning: pkg_resources",
            "Training until validation scores",
            "FutureWarning",
            "UserWarning",
            "Did not meet early stopping",
            "num_leaves is set=",
        )

        # --- Stream stdout ---
        log_entry = None
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue

            # Skip noisy repeated warnings
            if any(p in line for p in _NOISY_PATTERNS):
                continue

            level = "info"
            if "ERROR" in line or "Error" in line:
                level = "error"
            elif "WARNING" in line or "Warning" in line:
                level = "warning"
            elif "完成" in line or "success" in line.lower() or "✓" in line:
                level = "success"

            log_entry = {
                "id": _gen_id(),
                "timestamp": _now(),
                "level": level,
                "message": line[:500],
            }
            task["logs"].append(log_entry)
            if len(task["logs"]) > 2000:
                task["logs"] = task["logs"][-2000:]

            # Broadcast log to WebSocket
            await _broadcast(task_id, {
                "type": "log",
                "taskId": task_id,
                "data": log_entry,
                "timestamp": _now(),
            })

            # Update progress for meaningful lines
            if any(kw in line for kw in ["因子", "回测", "模型", "训练", "完成", "加载",
                                          "[1/4]", "[2/4]", "[3/4]", "[4/4]", "结果"]):
                task["progress"]["message"] = line[:200]

                # Estimate progress from run_backtest step markers
                if "[1/4]" in line:
                    task["progress"]["progress"] = 15
                elif "[2/4]" in line:
                    task["progress"]["progress"] = 35
                elif "[3/4]" in line:
                    task["progress"]["progress"] = 55
                elif "[4/4]" in line:
                    task["progress"]["progress"] = 75
                elif "结果已保存" in line or "回测结果" in line:
                    task["progress"]["progress"] = 95

                task["progress"]["timestamp"] = _now()
                await _broadcast(task_id, {
                    "type": "progress",
                    "taskId": task_id,
                    "data": task["progress"],
                    "timestamp": _now(),
                })

        # --- Process exit ---
        exit_code = await proc.wait()
        task["pid"] = None
        task["status"] = "completed" if exit_code == 0 else "failed"
        task["updatedAt"] = _now()

        # Try to load backtest results from output metrics JSON
        if exit_code == 0:
            task["progress"]["phase"] = "completed"
            task["progress"]["progress"] = 100
            task["progress"]["message"] = "回测完成"
            if not task.get("metrics"):
                _load_backtest_results(task)
        else:
            task["progress"]["message"] = f"回测失败 (exit code: {exit_code})"

        await _broadcast(task_id, {
            "type": "result",
            "taskId": task_id,
            "data": {
                "status": task["status"],
                "metrics": task.get("metrics", {}),
            },
            "timestamp": _now(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["progress"]["message"] = str(e)
        task["updatedAt"] = _now()
        await _broadcast(task_id, {
            "type": "error",
            "taskId": task_id,
            "data": {"error": str(e)},
            "timestamp": _now(),
        })


def _load_backtest_results(task: Dict[str, Any]):
    """Try to load backtest result metrics from the output directory."""
    try:
        config_path = task.get("config", {}).get("configPath") or str(
            PROJECT_ROOT / "configs" / "backtest.yaml"
        )
        with open(config_path, "r") as f:
            bt_config = yaml.safe_load(f)
        output_dir_raw = bt_config.get("experiment", {}).get(
            "output_dir", "data/results/backtest_v2_results"
        )
        # Resolve relative output_dir against PROJECT_ROOT (run_backtest runs with cwd=PROJECT_ROOT)
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir_str = str(output_dir)

        # Look for most recent metrics JSON
        metrics_files = sorted(
            glob.glob(os.path.join(output_dir_str, "*_backtest_metrics.json")),
            key=os.path.getmtime, reverse=True,
        )
        if metrics_files:
            with open(metrics_files[0], "r") as f:
                metrics_data = json.load(f)
            # The JSON has a nested structure: { metrics: {...}, config: {...}, ... }
            # Flatten: put the inner metrics dict at the top level for the frontend,
            # but also keep meta fields like experiment_name and elapsed_seconds.
            inner_metrics = metrics_data.get("metrics", {})
            flat = {**inner_metrics}
            # Carry over useful metadata
            for key in ("experiment_name", "factor_source", "num_factors",
                        "config", "elapsed_seconds"):
                if key in metrics_data:
                    flat[f"__{key}"] = metrics_data[key]
            
            # Load cumulative excess return data from CSV
            csv_path = metrics_files[0].replace("_backtest_metrics.json", "_cumulative_excess.csv")
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                if 'date' in df.columns and 'cumulative_excess_return' in df.columns:
                    cumulative_data = df[['date', 'cumulative_excess_return']].to_dict('records')
                    flat["cumulative_curve"] = [
                        {"date": r["date"], "value": r["cumulative_excess_return"]} 
                        for r in cumulative_data
                    ]

            task["metrics"] = flat
    except Exception as e:
        import traceback
        traceback.print_exc()  # print for debugging, but don't crash


# ---- System config endpoints ----

@app.get("/api/v1/system/config", response_model=ApiResponse)
async def get_system_config():
    """Read current system configuration from .env and experiment.yaml."""
    dotenv = _load_dotenv_dict()

    # Read experiment.yaml for display
    exp_yaml_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    exp_yaml_content = ""
    if exp_yaml_path.exists():
        exp_yaml_content = exp_yaml_path.read_text(encoding="utf-8")

    # Mask API keys for security
    masked_env = {}
    for k, v in dotenv.items():
        if "KEY" in k.upper() and v:
            masked_env[k] = v[:8] + "..." + v[-4:] if len(v) > 12 else "***"
        else:
            masked_env[k] = v

    return ApiResponse(
        success=True,
        data={
            "env": masked_env,
            "experimentYaml": exp_yaml_content,
            "factorLibraries": [Path(p).name for p in _displayable_factor_jsons()],
        },
    )


@app.put("/api/v1/system/config", response_model=ApiResponse)
async def update_system_config(update: SystemConfigUpdate):
    """Update .env configuration (non-secret fields only)."""
    if not DOTENV_PATH.exists():
        raise HTTPException(status_code=404, detail=".env file not found")

    content = DOTENV_PATH.read_text(encoding="utf-8")
    updates = {k: v for k, v in update.model_dump().items() if v is not None}

    import re
    for key, val in updates.items():
        # Replace existing line or append
        pattern = rf"^{re.escape(key)}\s*=.*$"
        replacement = f"{key}={val}"
        new_content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if n > 0:
            content = new_content
        else:
            content += f"\n{replacement}\n"

    DOTENV_PATH.write_text(content, encoding="utf-8")
    return ApiResponse(success=True, message="配置已更新")


# ---- WebSocket endpoint ----

@app.websocket("/ws/mining/{task_id}")
async def ws_mining(websocket: WebSocket, task_id: str):
    """WebSocket for real-time experiment updates."""
    await websocket.accept()

    if task_id not in ws_connections:
        ws_connections[task_id] = []
    ws_connections[task_id].append(websocket)

    # Send current state immediately
    if task_id in tasks:
        try:
            await websocket.send_json({
                "type": "progress",
                "taskId": task_id,
                "data": tasks[task_id].get("progress", {}),
                "timestamp": _now(),
            })
            # Send recent logs
            for log in tasks[task_id].get("logs", [])[-20:]:
                await websocket.send_json({
                    "type": "log",
                    "taskId": task_id,
                    "data": log,
                    "timestamp": _now(),
                })
        except Exception:
            pass

    try:
        while True:
            data = await websocket.receive_text()
            # Heartbeat
            if data == "ping":
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": _now(),
                })
    except WebSocketDisconnect:
        if task_id in ws_connections:
            try:
                ws_connections[task_id].remove(websocket)
            except ValueError:
                pass


# ========================== Entry Point ==========================

def _update_mining_metrics(task: Dict[str, Any]):
    """
    Update mining task metrics from the generated factor library.
    Calculates best factor stats and extracts top 10 factors.
    """
    jsons = _find_factor_jsons()
    # Prefer library with matching suffix if configured
    target_lib = None
    config = task.get("config", {})
    suffix = config.get("librarySuffix")
    
    if suffix:
        candidate = PROJECT_ROOT / "data" / "factorlib" / f"all_factors_library_{suffix}.json"
        # Fix: If suffix is specified, we ONLY look at this file.
        # If it doesn't exist yet, it means no factors have been mined yet for this task.
        if candidate.exists():
            target_lib = str(candidate)
        else:
            # Task specific file not found -> assume empty state
            return
            
    elif jsons:
        # No suffix provided, fallback to latest existing library (legacy behavior)
        target_lib = jsons[0]
        
    if not target_lib:
        return

    # Check modification time
    try:
        mtime = os.path.getmtime(target_lib)
        created_at_str = task.get("createdAt")
        if created_at_str:
            created_at_dt = datetime.fromisoformat(created_at_str)
            # Add a small buffer (e.g. 1 second) to avoid race conditions where file is created immediately
            if mtime < created_at_dt.timestamp():
                # File is older than the task -> ignore it
                return
    except Exception:
        pass

    try:
        lib = _load_factor_library(target_lib)
        factors = lib.get("factors", {})
        
        # 1. Update basic stats
        total = len(factors)
        task["metrics"]["totalFactors"] = total
        
        high = medium = low = 0
        factor_list = []
        
        for f_id, f_info in factors.items():
            # Check if this factor was created after task start
            # If we are using a shared library file (unlikely with new logic, but possible if user forces it),
            # we must ensure we don't display old factors.
            try:
                added_at_str = f_info.get("added_at", "")
                created_at_str = task.get("createdAt", "")
                if added_at_str and created_at_str:
                    # Parse timestamps
                    # added_at usually in isoformat
                    added_at_dt = datetime.fromisoformat(added_at_str)
                    created_at_dt = datetime.fromisoformat(created_at_str)
                    if added_at_dt < created_at_dt:
                        continue
            except Exception:
                pass # If date parsing fails, be permissive or conservative? Permissive for now.

            bt = f_info.get("test_backtest_metrics", f_info.get("backtest_results", {}))
            if not isinstance(bt, dict):
                bt = {}
            q = _classify_quality(f_info, bt)
            if q == "high": high += 1
            elif q == "medium": medium += 1
            else: low += 1

            # Prepare for top 10 list - extract metrics from test_backtest_metrics
            import math
            def _sn(v, d=0):
                if v is None: return d
                try:
                    fv = float(v)
                    return d if math.isnan(fv) or math.isinf(fv) else fv
                except: return d

            rank_ic = _sn(bt.get("rankic", bt.get("rank_ic", bt.get("Rank IC", 0))))
            rank_icir = _sn(bt.get("rankicir", bt.get("rank_ic_ir", bt.get("Rank ICIR", 0))))
            ic = _sn(bt.get("ic", bt.get("IC", rank_ic)))
            icir = _sn(bt.get("icir", bt.get("ICIR", rank_icir)))

            cumulative_curve = []
            annual_ret = _sn(bt.get("long_ret", bt.get("long_netret", bt.get("1day.excess_return_without_cost.annualized_return", 0))))
            max_dd = _sn(bt.get("long_maxdd", bt.get("long_netmaxdd", bt.get("1day.excess_return_with_cost.max_drawdown", 0))))
            
            # Calmar Ratio = Annual Return / Max Drawdown (absolute value)
            # Avoid division by zero
            cr = 0
            if max_dd < 0:
                cr = annual_ret / abs(max_dd)
            elif max_dd > 0:
                cr = annual_ret / max_dd
            
            # Simple simulation: 20 data points for preview sparkline
            import random
            current_val = 1.0
            # Daily drift approx
            drift = (1 + annual_ret) ** (1/252) - 1 if annual_ret else 0
            vol = 0.02 # Assumed daily vol
            
            # Use factor name hash to seed random for consistency
            random.seed(hash(f_info.get("factor_name", f_id)))
            
            for i in range(20):
                 # Generate last 20 points
                 ret = random.gauss(drift, vol)
                 current_val *= (1 + ret)
                 cumulative_curve.append({"value": current_val, "date": f"Day {i+1}"})
            
            factor_list.append({
                "factorName": f_info.get("factor_name", f_id),
                "factorExpression": f_info.get("factor_expression", ""),
                "rankIc": rank_ic,
                "rankIcir": rank_icir,
                "ic": ic,
                "icir": icir,
                "annualReturn": annual_ret,
                "sharpeRatio": _sn(bt.get("long_ir", bt.get("long_netir", bt.get("1day.excess_return_with_cost.information_ratio", 0)))),
                "maxDrawdown": max_dd,
                "calmarRatio": cr,
                "cumulativeCurve": cumulative_curve
            })

        task["metrics"]["highQualityFactors"] = high
        task["metrics"]["mediumQualityFactors"] = medium
        task["metrics"]["lowQualityFactors"] = low
        
        # 2. Find best factor
        if factor_list:
            # Sort by RankIC desc
            factor_list.sort(key=lambda x: x["rankIc"], reverse=True)
            best = factor_list[0]
            
            # Update task metrics with best factor's stats
            task["metrics"]["annualReturn"] = best["annualReturn"]
            task["metrics"]["rankIc"] = best["rankIc"]
            task["metrics"]["sharpeRatio"] = best["sharpeRatio"]
            task["metrics"]["maxDrawdown"] = best["maxDrawdown"]
            task["metrics"]["factorName"] = best["factorName"]
            
            # 3. Top 10 Factors
            task["metrics"]["top10Factors"] = factor_list[:10]
            
    except Exception:
        pass # Best effort

# ========================== Factor Analysis Endpoints ==========================


class AnalysisStartRequest(BaseModel):
    factorIds: List[str] = Field(..., description="Factor IDs to analyze")
    library: str = Field(..., description="Source factor library filename")


class AddToLibraryRequest(BaseModel):
    factorIds: List[str] = Field(..., description="Factor IDs to add")
    targetLibrary: str = Field(..., description="Target library filename")
    createNew: bool = Field(False, description="Create new library if true")


class FactorUpdateRequest(BaseModel):
    factorExpression: Optional[str] = None
    factorDescription: Optional[str] = None
    factorFormulation: Optional[str] = None


@app.post("/api/v1/analysis/start", response_model=ApiResponse)
async def start_analysis(req: AnalysisStartRequest):
    """Start factor analysis (tq diagnostics) for selected factors."""
    task_id = _gen_id()
    task = {
        "taskId": task_id,
        "status": "running",
        "type": "analysis",
        "config": req.model_dump(),
        "progress": {
            "phase": "analyzing",
            "currentRound": 0,
            "totalRounds": len(req.factorIds),
            "progress": 0,
            "message": "正在启动因子分析...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {},
        "result": None,
        "pid": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task
    asyncio.create_task(_run_analysis(task_id, req))
    return ApiResponse(success=True, data={"taskId": task_id, "task": task})


async def _run_analysis(task_id: str, req: AnalysisStartRequest):
    """Run tq factor analysis as a subprocess for each factor."""
    task = tasks[task_id]
    try:
        env = os.environ.copy()
        dotenv = _load_dotenv_dict()
        env.update(dotenv)

        # Resolve library path
        lib_path = PROJECT_ROOT / "data" / "factorlib" / req.library
        if not lib_path.exists():
            lib_path = PROJECT_ROOT / req.library

        # Load factor info
        lib_data = _load_factor_library(str(lib_path))
        factors_dict = lib_data.get("factors", {})

        analyzed = []
        total = len(req.factorIds)

        for idx, fid in enumerate(req.factorIds):
            factor_info = factors_dict.get(fid, {})
            factor_name = factor_info.get("factor_name", fid)
            factor_expr = factor_info.get("factor_expression", "")

            task["progress"]["currentRound"] = idx + 1
            task["progress"]["progress"] = int((idx / total) * 100)
            task["progress"]["message"] = f"正在分析因子: {factor_name} ({idx+1}/{total})"
            task["progress"]["timestamp"] = _now()

            await _broadcast(task_id, {
                "type": "progress",
                "taskId": task_id,
                "data": task["progress"],
                "timestamp": _now(),
            })

            log_entry = {
                "id": _gen_id(),
                "timestamp": _now(),
                "level": "info",
                "message": f"开始分析因子: {factor_name}",
            }
            task["logs"].append(log_entry)
            await _broadcast(task_id, {"type": "log", "taskId": task_id, "data": log_entry, "timestamp": _now()})

            # Run TQ backtest with diagnostics plot
            analysis_out = PROJECT_ROOT / "data" / "tq_upstream" / f"analysis_{task_id}" / factor_name
            analysis_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "-m", "quantaalpha.backtest.run_backtest",
                "--library_path", str(lib_path),
                "--library_factor", fid,
                "--plot", "True",
                "--quality_report", "True",
                "--output_dir", str(analysis_out),
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )

            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                level = "info"
                if "ERROR" in line:
                    level = "error"
                elif "WARNING" in line:
                    level = "warning"
                elif "完成" in line or "success" in line.lower():
                    level = "success"
                log_entry = {"id": _gen_id(), "timestamp": _now(), "level": level, "message": line[:500]}
                task["logs"].append(log_entry)
                if len(task["logs"]) > 1000:
                    task["logs"] = task["logs"][-1000:]
                await _broadcast(task_id, {"type": "log", "taskId": task_id, "data": log_entry, "timestamp": _now()})

            exit_code = await proc.wait()
            if exit_code == 0:
                analyzed.append(factor_name)
                log_entry = {"id": _gen_id(), "timestamp": _now(), "level": "success", "message": f"因子 {factor_name} 分析完成"}
            else:
                log_entry = {"id": _gen_id(), "timestamp": _now(), "level": "error", "message": f"因子 {factor_name} 分析失败 (exit: {exit_code})"}
            task["logs"].append(log_entry)
            await _broadcast(task_id, {"type": "log", "taskId": task_id, "data": log_entry, "timestamp": _now()})

        task["status"] = "completed"
        task["progress"]["phase"] = "completed"
        task["progress"]["progress"] = 100
        task["progress"]["message"] = f"分析完成: {len(analyzed)}/{total} 个因子"
        task["metrics"] = {"analyzed": analyzed, "total": total}
        task["updatedAt"] = _now()

        await _broadcast(task_id, {
            "type": "result",
            "taskId": task_id,
            "data": {"status": "completed", "metrics": task["metrics"]},
            "timestamp": _now(),
        })

    except Exception as e:
        task["status"] = "failed"
        task["progress"]["message"] = str(e)
        task["updatedAt"] = _now()
        await _broadcast(task_id, {"type": "error", "taskId": task_id, "data": {"error": str(e)}, "timestamp": _now()})


@app.get("/api/v1/analysis/images/{factor_name}")
async def get_analysis_image(factor_name: str):
    """Serve the diagnostics PNG for a factor from data/tq_upstream/."""
    from fastapi.responses import FileResponse

    # Search for the diagnostic PNG
    tq_dir = PROJECT_ROOT / "data" / "tq_upstream"
    # Look in all experiment subdirectories
    for exp_dir in sorted(tq_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if tq_dir.exists() else []:
        if not exp_dir.is_dir():
            continue
        png_path = exp_dir / factor_name / f"{factor_name}_diagnostics.png"
        if png_path.exists():
            return FileResponse(str(png_path), media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Diagnostics image not found for: {factor_name}")


@app.post("/api/v1/factors/add-to-library", response_model=ApiResponse)
async def add_factors_to_library(req: AddToLibraryRequest):
    """Add selected factors to a target library (existing or new)."""
    factorlib_dir = PROJECT_ROOT / "data" / "factorlib"
    factorlib_dir.mkdir(parents=True, exist_ok=True)

    target_path = factorlib_dir / req.targetLibrary
    if req.createNew or not target_path.exists():
        # Create new library
        target_data = {"metadata": {"total_factors": 0, "last_updated": _now()}, "factors": {}}
    else:
        target_data = _load_factor_library(str(target_path))

    # Find factors from all libraries
    all_jsons = _find_factor_jsons()
    added = 0
    for fid in req.factorIds:
        for lib_path in all_jsons:
            try:
                lib = _load_factor_library(lib_path)
                if fid in lib.get("factors", {}):
                    target_data["factors"][fid] = lib["factors"][fid]
                    added += 1
                    break
            except Exception:
                continue

    target_data["metadata"]["total_factors"] = len(target_data["factors"])
    target_data["metadata"]["last_updated"] = _now()

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)

    return ApiResponse(success=True, data={"added": added}, message=f"已添加 {added} 个因子到 {req.targetLibrary}")


@app.put("/api/v1/factors/{factor_id}", response_model=ApiResponse)
async def update_factor(factor_id: str, req: FactorUpdateRequest):
    """Update a factor's expression, description, or formulation in its library.
    If expression changes, re-run single-factor backtest to update metrics."""
    all_jsons = _find_factor_jsons()
    for lib_path in all_jsons:
        try:
            lib = _load_factor_library(lib_path)
            if factor_id in lib.get("factors", {}):
                factor = lib["factors"][factor_id]
                expression_changed = (
                    req.factorExpression is not None
                    and req.factorExpression != factor.get("factor_expression")
                )
                if req.factorExpression is not None:
                    factor["factor_expression"] = req.factorExpression
                if req.factorDescription is not None:
                    factor["factor_description"] = req.factorDescription
                if req.factorFormulation is not None:
                    factor["factor_formulation"] = req.factorFormulation
                factor["last_modified"] = _now()

                re_eval_result = None
                if expression_changed:
                    try:
                        from quantaalpha.api.services.library_service import LibraryService
                        svc = LibraryService()
                        re_eval_result = svc._re_evaluate_factor(
                            factor_id=factor_id,
                            factor_name=factor.get("factor_name", factor_id),
                            expression=req.factorExpression,
                        )
                        if re_eval_result.get("test_backtest_metrics"):
                            factor["test_backtest_metrics"] = re_eval_result["test_backtest_metrics"]
                        if re_eval_result.get("train_check_flags"):
                            factor["train_check_flags"] = re_eval_result["train_check_flags"]
                        if re_eval_result.get("test_check_flags"):
                            factor["test_check_flags"] = re_eval_result["test_check_flags"]
                        if re_eval_result.get("style_exposures") is not None:
                            factor["style_exposures"] = re_eval_result["style_exposures"]
                    except Exception as e:
                        import traceback
                        traceback.print_exc()

                with open(lib_path, "w", encoding="utf-8") as f:
                    json.dump(lib, f, ensure_ascii=False, indent=2)

                # Reclassify quality after potential re-evaluation
                bt = factor.get("test_backtest_metrics", {})
                if not isinstance(bt, dict):
                    bt = {}
                quality = _classify_quality(factor, bt)

                return ApiResponse(
                    success=True,
                    message="因子已更新" + (" (已重新回测)" if expression_changed and re_eval_result else ""),
                    data={"quality": quality, "re_evaluated": bool(expression_changed and re_eval_result)},
                )
        except Exception:
            import traceback
            traceback.print_exc()
            continue
    raise HTTPException(status_code=404, detail="Factor not found")


@app.delete("/api/v1/factors/libraries/{library_name}", response_model=ApiResponse)
async def delete_factor_library(library_name: str):
    """Delete an entire factor library JSON file."""
    factorlib_dir = PROJECT_ROOT / "data" / "factorlib"
    lib_path = factorlib_dir / library_name
    if not lib_path.exists() or not lib_path.is_file():
        raise HTTPException(status_code=404, detail=f"因子库 {library_name} 不存在")
    if not lib_path.name.endswith(".json"):
        raise HTTPException(status_code=400, detail="只能删除 JSON 因子库文件")
    try:
        data = json.loads(lib_path.read_text(encoding="utf-8"))
        factor_count = len(data.get("factors", {}))
        lib_path.unlink()
        return ApiResponse(success=True, message=f"已删除因子库 {library_name}（含 {factor_count} 个因子）")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/factors/{factor_id}", response_model=ApiResponse)
async def delete_factor(factor_id: str, library: Optional[str] = Query(None)):
    """Delete a factor from its library JSON file."""
    all_jsons = _find_factor_jsons()
    if library:
        search_order = [p for p in all_jsons if Path(p).name == library]
        search_order += [p for p in all_jsons if Path(p).name != library]
    else:
        search_order = list(all_jsons)

    for lib_path in search_order:
        try:
            lib = _load_factor_library(lib_path)
            factors = lib.get("factors", {})
            if factor_id in factors:
                del factors[factor_id]
                lib["factors"] = factors
                if "metadata" in lib:
                    lib["metadata"]["total_factors"] = len(factors)
                    lib["metadata"]["last_updated"] = _now()
                with open(lib_path, "w", encoding="utf-8") as f:
                    json.dump(lib, f, ensure_ascii=False, indent=2)
                return ApiResponse(
                    success=True,
                    message=f"因子 {factor_id} 已从 {Path(lib_path).name} 中删除",
                )
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="Factor not found")


# ========================== ML Backtest Endpoints ==========================


class MLBacktestStartRequest(BaseModel):
    model: str = Field("lgbm", description="Model type: lgbm or xgb")
    optuna_trials: int = Field(20, description="Optuna hyperparameter search trials")
    label_type: str = Field("return_regression", description="Label type")
    split_mode: str = Field("walk_forward", description="Split mode: walk_forward, rolling, or fixed")
    rolling_window: int = Field(504, description="Rolling window size in trading days")
    train_lookback: Optional[int] = Field(None, description="Walk-forward train lookback in trading days")
    validation_size: int = Field(63, description="Validation set size in trading days")
    gap_size: int = Field(1, description="Gap between val and test in trading days (anti-leakage)")
    test_size: int = Field(63, description="Test set size in trading days")
    top_n: int = Field(50, description="Number of stocks selected each prediction day")
    universe: str = Field("hs300", description="Stock universe: hs300, zz500, zz800, zz1000, sz50")
    walk_test_start: Optional[str] = Field(None, description="Walk-forward test period start date")
    walk_test_end: Optional[str] = Field(None, description="Walk-forward test period end date")
    train_start: Optional[str] = Field(None, description="Fixed mode: train start date (YYYY-MM-DD)")
    train_end: Optional[str] = Field(None, description="Fixed mode: train end date")
    val_start: Optional[str] = Field(None, description="Fixed mode: validation start date")
    val_end: Optional[str] = Field(None, description="Fixed mode: validation end date")
    test_start: Optional[str] = Field(None, description="Fixed mode: test start date")
    test_end: Optional[str] = Field(None, description="Fixed mode: test end date")
    factor_library: str = Field(..., description="Factor library filename")
    quality_filter: str = Field("all", description="Quality filter: high/medium/all")


def _build_ml_backtest_command(req: MLBacktestStartRequest, lib_path: Path) -> List[str]:
    cmd = [
        sys.executable, "-m", "quantaalpha.backtest.run_ml_backtest",
        "--factor-json", str(lib_path),
        "--model", req.model,
        "--optuna-trials", str(req.optuna_trials),
        "--label-type", req.label_type,
        "--rolling-window", str(req.rolling_window),
        "--validation-size", str(req.validation_size),
        "--gap-size", str(req.gap_size),
        "--test-size", str(req.test_size),
        "--top-n", str(req.top_n),
        "--universe", req.universe,
        "--quality-filter", req.quality_filter,
        "--skip-uncached",
        "-v",
    ]
    if req.split_mode == "walk_forward" and req.walk_test_start and req.walk_test_end:
        cmd += [
            "--split-mode", "walk_forward",
            "--train-lookback", str(req.train_lookback or req.rolling_window),
            "--walk-test-start", req.walk_test_start,
            "--walk-test-end", req.walk_test_end,
        ]
    if req.split_mode == "fixed" and req.train_start:
        cmd += [
            "--split-mode", "fixed",
            "--train-start", req.train_start,
            "--train-end", req.train_end or "",
            "--val-start", req.val_start or "",
            "--val-end", req.val_end or "",
            "--test-start", req.test_start or "",
            "--test-end", req.test_end or "",
        ]
    return cmd


def _ingest_ml_result_line(task: Dict[str, Any], line: str) -> bool:
    prefix = "__ML_RESULT__:"
    if not line.startswith(prefix):
        return False
    try:
        metrics = json.loads(line[len(prefix):])
    except Exception:
        return False
    if isinstance(metrics, dict):
        task["metrics"] = metrics
        task["result"] = metrics
        return True
    return False


@app.post("/api/v1/ml-backtest/start", response_model=ApiResponse)
async def start_ml_backtest(req: MLBacktestStartRequest):
    """Start ML model backtest with rolling window."""
    task_id = _gen_id()
    task = {
        "taskId": task_id,
        "status": "running",
        "type": "ml_backtest",
        "config": req.model_dump(),
        "progress": {
            "phase": "backtesting",
            "currentRound": 0,
            "totalRounds": 1,
            "progress": 0,
            "message": "正在启动ML回测...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {},
        "result": None,
        "pid": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task
    asyncio.create_task(_run_ml_backtest(task_id, req))
    return ApiResponse(success=True, data={"taskId": task_id, "task": task})


@app.get("/api/v1/ml-backtest/{task_id}", response_model=ApiResponse)
async def get_ml_backtest_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data={"task": tasks[task_id]})


@app.delete("/api/v1/ml-backtest/{task_id}", response_model=ApiResponse)
async def cancel_ml_backtest(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.get("pid"):
        try:
            os.kill(task["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    task["status"] = "cancelled"
    task["updatedAt"] = _now()
    await _broadcast(task_id, {"type": "result", "taskId": task_id, "data": {"status": "cancelled"}, "timestamp": _now()})
    return ApiResponse(success=True, message="ML回测已取消")


async def _run_ml_backtest(task_id: str, req: MLBacktestStartRequest):
    """Run ML backtest subprocess."""
    task = tasks[task_id]
    try:
        env = os.environ.copy()
        dotenv = _load_dotenv_dict()
        env.update(dotenv)

        # Resolve factor library path
        factor_json = req.factor_library
        lib_path = PROJECT_ROOT / "data" / "factorlib" / factor_json
        if not lib_path.exists():
            lib_path = PROJECT_ROOT / factor_json

        cmd = _build_ml_backtest_command(req, lib_path)

        applied_log = {
            "id": _gen_id(),
            "timestamp": _now(),
            "level": "info",
            "message": (
                "Applied ML backtest settings: "
                f"model={req.model}, optuna_trials={req.optuna_trials}, "
                f"label_type={req.label_type}, split_mode={req.split_mode}, "
                f"train_lookback={req.train_lookback or req.rolling_window}, "
                f"validation_size={req.validation_size}, gap_size={req.gap_size}, "
                f"test_size={req.test_size}, top_n={req.top_n}, "
                f"walk_test_start={req.walk_test_start}, walk_test_end={req.walk_test_end}, "
                f"factor_library={Path(lib_path).name}, quality_filter={req.quality_filter}"
            ),
        }
        task["logs"].append(applied_log)
        await _broadcast(task_id, {"type": "log", "taskId": task_id, "data": applied_log, "timestamp": _now()})

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        task["pid"] = proc.pid

        _NOISY = (
            "field data contains nan", "common_infra", "PyTorch models are skipped",
            "UserWarning", "FutureWarning", "Training until validation scores",
            "Did not meet early stopping", "num_leaves is set=",
        )

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line or any(p in line for p in _NOISY):
                continue

            if _ingest_ml_result_line(task, line):
                await _broadcast(task_id, {
                    "type": "result",
                    "taskId": task_id,
                    "data": {"status": "completed", "metrics": task.get("metrics", {})},
                    "timestamp": _now(),
                })
                continue

            level = "info"
            if "ERROR" in line:
                level = "error"
            elif "WARNING" in line:
                level = "warning"
            elif "完成" in line or "success" in line.lower():
                level = "success"

            log_entry = {"id": _gen_id(), "timestamp": _now(), "level": level, "message": line[:500]}
            task["logs"].append(log_entry)
            if len(task["logs"]) > 2000:
                task["logs"] = task["logs"][-2000:]

            await _broadcast(task_id, {"type": "log", "taskId": task_id, "data": log_entry, "timestamp": _now()})

            # Progress estimation
            if "滚动窗口" in line or "window" in line.lower():
                task["progress"]["message"] = line[:200]
                task["progress"]["timestamp"] = _now()
                await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": task["progress"], "timestamp": _now()})

        exit_code = await proc.wait()
        task["pid"] = None
        task["status"] = "completed" if exit_code == 0 else "failed"
        task["updatedAt"] = _now()

        if exit_code == 0:
            task["progress"]["phase"] = "completed"
            task["progress"]["progress"] = 100
            task["progress"]["message"] = "ML回测完成"
            _load_backtest_results(task)
        else:
            task["progress"]["message"] = f"ML回测失败 (exit code: {exit_code})"

        await _broadcast(task_id, {
            "type": "result", "taskId": task_id,
            "data": {"status": task["status"], "metrics": task.get("metrics", {})},
            "timestamp": _now(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["progress"]["message"] = str(e)
        task["updatedAt"] = _now()
        await _broadcast(task_id, {"type": "error", "taskId": task_id, "data": {"error": str(e)}, "timestamp": _now()})


# ========================== Static File Serving ==========================

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse as _FileResponse

# Serve tq_upstream PNG results as static files
_tq_upstream_dir = PROJECT_ROOT / "data" / "tq_upstream"
if _tq_upstream_dir.exists():
    app.mount("/static/tq_upstream", StaticFiles(directory=str(_tq_upstream_dir)), name="tq_upstream")

# ---------------------------------------------------------------------------
# Frontend SPA: serve .build/ directory (Vite output)
# ---------------------------------------------------------------------------
_build_dir = Path(__file__).resolve().parent.parent / ".build"
if _build_dir.exists():
    # Mount assets sub-directory for JS/CSS bundles
    _assets_dir = _build_dir / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend_assets")

    # Catch-all: serve index.html for any non-API route (SPA routing)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # If the path points to an existing file in .build/, serve it
        file_path = _build_dir / full_path
        if full_path and file_path.exists() and file_path.is_file():
            import mimetypes
            mime, _ = mimetypes.guess_type(str(file_path))
            return _FileResponse(str(file_path), media_type=mime or "application/octet-stream")
        # Otherwise serve index.html for SPA client-side routing
        index_path = _build_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return {"message": "QuantaAlpha API", "version": "2.0.0"}


# ========================== Entry Point ==========================

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
