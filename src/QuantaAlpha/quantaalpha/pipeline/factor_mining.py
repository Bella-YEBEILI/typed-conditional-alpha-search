"""
Factor workflow with session control and evolution support.

Supports three round phases:
- Original: Initial exploration in each direction
- Mutation: Orthogonal exploration from parent trajectories
- Crossover: Hybrid strategies from multiple parents

Supports parallel execution within each phase when enabled.
"""

from typing import Any
from pathlib import Path
import fire
import signal
import sys
import threading
from multiprocessing import Process, Queue
from functools import wraps
import time
import ctypes
import os
import pickle
from quantaalpha.backtest import configure_backtest_engine, load_backtest_config, resolve_backtest_engine
from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING
from quantaalpha.pipeline.planning import generate_parallel_directions
from quantaalpha.pipeline.planning import load_run_config
from quantaalpha.pipeline.loop import AlphaAgentLoop
from quantaalpha.pipeline.evolution import (
    EvolutionController, 
    EvolutionConfig,
    StrategyTrajectory,
    RoundPhase,
)
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.log import logger
from quantaalpha.log.time import measure_time
from quantaalpha.llm.config import LLM_SETTINGS
from quantaalpha.factors.proposal import FactorHypothesis, AlphaAgentHypothesis
from quantaalpha.factors.experiment import FactorMiningExperiment, FactorTask
from quantaalpha.factors.data_domains import (
    describe_factor_domains,
    normalize_factor_mode,
    resolve_effective_factor_domains,
)
from quantaalpha.core.proposal import HypothesisFeedback
import json
import random

DIRECTION_FILE_CANDIDATES = {
    "pv": (
        "original_direction_daily.json",
        "original_direction_pv.json",
        "original_direction.json",
    ),
    "fundamental": (
        "original_direction_cross_sectional.json",
        "original_direction_fundamental.json",
        "original_direction.json",
    ),
    "minutes": (
        "original_direction_minutes.json",
        "original_direction_minute.json",
        "original_direction.json",
    ),
    "joint": (
        "original_direction_joint.json",
        "original_direction.json",
    ),
}


def _has_trajectory_output(traj_data: dict[str, Any]) -> bool:
    """Return True only when a loop produced a complete, evolvable trajectory."""
    if not isinstance(traj_data, dict):
        return False
    if traj_data.get("hypothesis") is None or traj_data.get("feedback") is None:
        return False
    experiment = traj_data.get("experiment")
    if experiment is None:
        return False
    sub_tasks = getattr(experiment, "sub_tasks", None)
    if sub_tasks is None:
        sub_tasks = getattr(experiment, "tasks", None)
    return bool(sub_tasks)


def _task_attempt_key(task: dict[str, Any]) -> tuple[str, int, int, str, tuple[str, ...]]:
    phase = task.get("phase")
    phase_value = phase.value if hasattr(phase, "value") else str(phase)
    parent_ids = tuple(
        sorted(
            str(getattr(parent, "trajectory_id", "") or "")
            for parent in (task.get("parent_trajectories") or [])
            if str(getattr(parent, "trajectory_id", "") or "")
        )
    )
    intent = str(task.get("mutation_intent") or task.get("crossover_intent") or "")
    return (
        phase_value,
        int(task.get("round_idx", 0) or 0),
        int(task.get("direction_id", 0) or 0),
        intent,
        parent_ids,
    )


def _record_empty_task_attempt(
    attempts: dict[tuple[str, int, int, str, tuple[str, ...]], int],
    task: dict[str, Any],
    max_retries: int,
) -> bool:
    """Record an empty task result and return True when it should be retried."""
    key = _task_attempt_key(task)
    attempts[key] = attempts.get(key, 0) + 1
    return attempts[key] < max(1, int(max_retries))


def _format_trajectory_metric_profile(trajectory: StrategyTrajectory) -> str:
    """Summarize metric details without collapsing them into a single score."""
    metrics = {
        str(key): value
        for key, value in (trajectory.backtest_metrics or {}).items()
        if value is not None
    }
    if metrics:
        metrics_text = ", ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())
    else:
        metrics_text = "none"
    failed_metrics = ", ".join(trajectory.get_failed_metrics()) or "none"
    return (
        f"check_passed={trajectory.is_check_passed()}, "
        f"passed_metrics={len(trajectory.get_pass_set())}, "
        f"failed_metrics={failed_metrics}, "
        f"metrics={metrics_text}"
    )


def _parent_has_evolution_context(trajectory: StrategyTrajectory) -> bool:
    """Parent history should include failed-but-material trajectories, not only winners."""
    return bool(
        trajectory
        and getattr(trajectory, "factors", None)
        and (getattr(trajectory, "feedback", None) or getattr(trajectory, "submission_checks", None))
    )


def _load_direction_portfolios(json_path: str | Path) -> list[dict[str, Any]]:
    path = Path(json_path)
    if not path.exists():
        logger.warning(f"Direction file not found: {path}")
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        portfolios = data.get("factor_portfolios", [])
        if not portfolios:
            logger.warning(f"Direction file is empty: {path}")
            return []
        return portfolios
    except Exception as e:
        logger.error(f"Failed to load direction file {path}: {e}")
        return []


def _pick_random_initial_direction(
    json_path: str | Path,
    domain_label: str | None = None,
    used_identities: set[str] | None = None,
) -> str | None:
    """Randomly select one unseen seed direction from a JSON direction pool."""
    portfolios = _load_direction_portfolios(json_path)
    if not portfolios:
        return None

    available: list[str] = []
    seen = used_identities if used_identities is not None else set()
    for item in portfolios:
        desc = str(item.get("description", "")).strip()
        identity = _normalize_direction_identity(desc)
        if not desc or not identity or identity in seen:
            continue
        available.append(desc)

    if not available:
        logger.warning(f"No unseen direction left in {Path(json_path).name} for domain={domain_label}")
        return None

    desc = random.choice(available)
    if used_identities is not None:
        used_identities.add(_normalize_direction_identity(desc))
    domain_prefix = f"[{domain_label}] " if domain_label else ""
    logger.info(f"Selected {domain_prefix}random direction from {Path(json_path).name}: {desc[:80]}...")
    return desc


def _build_domain_seed_direction(
    experiment_dir: Path,
    data_domains: tuple[str, ...],
    used_direction_identities: dict[str, set[str]] | None = None,
) -> str | None:
    active_domains = tuple(dict.fromkeys(data_domains or ("pv",)))
    selected_segments: list[tuple[str, str, str]] = []
    used_map = used_direction_identities if used_direction_identities is not None else {}

    if set(active_domains) == {"pv", "minutes"}:
        candidate_files = DIRECTION_FILE_CANDIDATES["joint"]
        seen = used_map.setdefault("joint", set())
        for filename in candidate_files:
            selected_desc = _pick_random_initial_direction(
                experiment_dir / filename,
                domain_label="joint",
                used_identities=seen,
            )
            if selected_desc:
                return selected_desc
        logger.warning(
            f"No usable seed direction found for domain=joint. Tried files: {', '.join(candidate_files)}"
        )
        return None

    for domain in active_domains:
        candidate_files = DIRECTION_FILE_CANDIDATES.get(domain, ("original_direction.json",))
        selected_desc = None
        selected_file = None
        for filename in candidate_files:
            candidate_path = experiment_dir / filename
            seen = used_map.setdefault(domain, set())
            selected_desc = _pick_random_initial_direction(
                candidate_path,
                domain_label=domain,
                used_identities=seen,
            )
            if selected_desc:
                selected_file = filename
                break

        if not selected_desc:
            logger.warning(
                f"No usable seed direction found for domain={domain}. Tried files: {', '.join(candidate_files)}"
            )
            continue

        selected_segments.append((domain, selected_file or "", selected_desc))

    if not selected_segments:
        return None

    if len(selected_segments) == 1:
        return selected_segments[0][2]

    lines = ["Round-specific multi-domain seed directions:"]
    for domain, filename, desc in selected_segments:
        lines.append(f"[{domain} | source={filename}] {desc}")
    return "\n\n".join(lines)


def _merge_seed_direction(
    base_direction: str | None,
    seed_direction: str | None,
    context_label: str,
) -> str | None:
    """Merge one random seed direction into the base direction for a single experiment."""
    if not seed_direction:
        return base_direction

    base_text = str(base_direction or "").strip()
    if not base_text:
        logger.info(f"Injected {context_label} seed as the initial direction.")
        return seed_direction

    logger.info(f"Injected {context_label} seed ahead of existing branch context.")
    return "\n\n".join(
        [
            f"{context_label.capitalize()} initial direction:",
            seed_direction,
            "Existing branch context:",
            base_text,
        ]
    )


def _normalize_direction_identity(direction: str | None) -> str:
    return " ".join(str(direction or "").strip().lower().split())


def _dedupe_direction_list(directions: list[str | None]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for direction in directions:
        text = str(direction or "").strip()
        if not text:
            continue
        identity = _normalize_direction_identity(text)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(text)
    return unique


def _new_seed_tracker() -> dict[str, set[str]]:
    return {domain: set() for domain in DIRECTION_FILE_CANDIDATES}

def resolve_factor_modes(factor_cfg: dict, current_env: dict | None = None) -> tuple[str, str, str, tuple[str, ...]]:
    """Resolve data/prompt/backtest mode and active data domains from config/env."""
    env = current_env or os.environ

    data_mode_raw = env.get("FACTOR_DATA_MODE") or (factor_cfg or {}).get("data_mode", "daily")
    data_mode = normalize_factor_mode(data_mode_raw)
    if data_mode is None:
        logger.warning(f"Invalid factor.data_mode={data_mode_raw}, fallback to daily")
        data_mode = "daily"

    prompt_mode_raw = str(env.get("FACTOR_PROMPT_MODE") or (factor_cfg or {}).get("prompt_mode", "auto")).strip().lower()
    if prompt_mode_raw == "auto":
        prompt_mode = data_mode
    else:
        prompt_mode = normalize_factor_mode(prompt_mode_raw)
        if prompt_mode is None:
            logger.warning(f"Invalid factor.prompt_mode={prompt_mode_raw}, fallback to {data_mode}")
            prompt_mode = data_mode

    backtest_mode = str(env.get("FACTOR_BACKTEST_MODE") or (factor_cfg or {}).get("backtest_mode", "single")).strip().lower()
    if backtest_mode not in {"single", "combined", "hypothesis", "hypothesis_combined"}:
        logger.warning(f"Invalid factor.backtest_mode={backtest_mode}, fallback to single")
        backtest_mode = "single"

    raw_domains = env.get("FACTOR_DATA_DOMAINS")
    if raw_domains is None and isinstance(factor_cfg, dict):
        raw_domains = factor_cfg.get("data_domains")
    data_domains = resolve_effective_factor_domains(raw_domains, data_mode=data_mode)

    return data_mode, prompt_mode, backtest_mode, data_domains


def force_timeout():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            seconds = LLM_SETTINGS.factor_mining_timeout
            def handle_timeout(signum, frame):
                logger.error(f"Process terminated: timeout exceeded ({seconds}s)")
                sys.exit(1)

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


def _run_branch(
    direction: str | None,
    step_n: int,
    use_local: bool,
    idx: int,
    log_root: str,
    log_prefix: str,
    quality_gate_cfg: dict = None,
):
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=None,
        use_local=use_local,
        quality_gate_config=quality_gate_cfg or {},
    )
    model_loop.user_initial_direction = direction
    model_loop.run(step_n=step_n, stop_event=None)


def _run_evolution_task(
    task: dict[str, Any],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    stop_event: threading.Event | None,
    quality_gate_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single evolution task (one small loop).

    Args:
        task: Evolution task descriptor
        directions: List of original directions
        step_n: Steps per round
        use_local: Use local backtest
        user_direction: User initial direction
        log_root: Log root directory
        stop_event: Stop event
        quality_gate_cfg: Quality gate config

    Returns:
        Dict containing trajectory data
    """
    phase = task["phase"]
    direction_id = task["direction_id"]
    strategy_suffix = task.get("strategy_suffix", "")
    round_idx = task["round_idx"]
    parent_trajectories = task.get("parent_trajectories", [])
    crossover_intent = str(task.get("crossover_intent") or "")
    
    # Resolve direction by phase
    if phase == RoundPhase.ORIGINAL:
        direction = directions[direction_id] if direction_id < len(directions) else None
    elif phase == RoundPhase.MUTATION:
        direction = directions[direction_id] if direction_id < len(directions) else None
    else:  # CROSSOVER
        direction = None

    trajectory_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase)
    parent_ids = [p.trajectory_id for p in parent_trajectories]

    # Convert parent trajectories to trace history
    initial_history = []
    if parent_trajectories:
        try:
            for traj in parent_trajectories:
                if not _parent_has_evolution_context(traj):
                    continue
                
                # 1. Hypothesis
                hyp_details = traj.hypothesis_details or {}
                hyp = AlphaAgentHypothesis(
                    hypothesis=traj.hypothesis,
                    concise_observation=hyp_details.get("concise_observation", ""),
                    concise_justification=hyp_details.get("concise_justification", ""),
                    concise_knowledge=hyp_details.get("concise_knowledge", ""),
                    concise_specification=hyp_details.get("concise_specification", "")
                )
                
                # 2. Experiment
                tasks = []
                for f in traj.factors:
                    factor_expression = str(
                        f.get("factor_expression") or f.get("expression") or ""
                    )
                    factor_task = FactorTask(
                        factor_name=f.get("factor_name") or f.get("name", ""),
                        factor_description=f.get("factor_description") or f.get("description", ""),
                        factor_formulation=f.get("factor_formulation") or f.get("formulation", ""),
                        variables=f.get("variables", {}),
                    )
                    # Upstream rdagent FactorTask does not accept factor_expression in
                    # __init__, but QuantaAlpha coders/readers expect the attribute.
                    factor_task.factor_expression = factor_expression
                    tasks.append(factor_task)
                exp = FactorMiningExperiment(sub_tasks=tasks)
                result_factor_name = (
                    tasks[0].factor_name
                    if len(tasks) == 1 and hasattr(tasks[0], "factor_name")
                    else "SOTA Result"
                )
                exp.result = {result_factor_name: dict(traj.backtest_metrics or {})}
                
                # 3. Feedback
                fb_details = traj.feedback_details or {}
                fb = HypothesisFeedback(
                    observations=fb_details.get("observations", ""),
                    hypothesis_evaluation=fb_details.get("hypothesis_evaluation", ""),
                    new_hypothesis=fb_details.get("new_hypothesis", ""),
                    reason=fb_details.get("reason", ""),
                    decision=fb_details.get("decision", True),
                )
                
                initial_history.append((hyp, exp, fb))
            logger.info(f"Loaded {len(initial_history)} parent trajectories into trace history.")
        except Exception as e:
            logger.warning(f"Failed to convert parent trajectories to history: {e}")

    logger.info(f"Starting evolution task: phase={phase.value}, round={round_idx}, direction={direction_id}")

    # Create and run loop
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=stop_event,
        use_local=use_local,
        strategy_suffix=strategy_suffix,
        evolution_phase=phase.value,
        crossover_intent=crossover_intent,
        trajectory_id=trajectory_id,
        parent_trajectory_ids=parent_ids,
        initial_trace_history=initial_history,
        direction_id=direction_id,
        round_idx=round_idx,
        quality_gate_config=quality_gate_cfg or {},
    )
    model_loop.user_initial_direction = user_direction
    
    # Run one small loop (5 steps)
    model_loop.run(step_n=step_n, stop_event=stop_event)

    traj_data = model_loop._get_trajectory_data()
    traj_data["task"] = task
    if not _has_trajectory_output(traj_data):
        logger.warning(
            "Evolution task produced no usable trajectory output: "
            f"phase={phase.value}, round={round_idx}, direction={direction_id}"
        )
    
    return traj_data


def _parallel_task_worker(
    task: dict[str, Any],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    quality_gate_cfg: dict[str, Any] | None,
    result_queue: Queue,
    task_idx: int,
):
    """
    Worker for parallel evolution tasks. Runs one evolution task in a separate process and puts result in queue.
    Args: task, directions, step_n, use_local, user_direction, log_root, result_queue, task_idx.
    """
    try:
        from quantaalpha.core.conf import RD_AGENT_SETTINGS
        from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory
        RD_AGENT_SETTINGS.use_file_lock = False
        base_cache_dir = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str)
        base_cache_dir.mkdir(parents=True, exist_ok=True)
        task_cache_namespace = f"task_{task_idx}"
        task_workspace_dir = Path(RD_AGENT_SETTINGS.workspace_path) / f"task_{task_idx}"
        os.environ["PICKLE_CACHE_FOLDER_PATH_STR"] = str(base_cache_dir)
        os.environ["PICKLE_CACHE_NAMESPACE"] = task_cache_namespace
        os.environ["WORKSPACE_PATH"] = str(task_workspace_dir)
        RD_AGENT_SETTINGS.pickle_cache_folder_path_str = str(base_cache_dir)
        RD_AGENT_SETTINGS.workspace_path = task_workspace_dir

        if task.get("parent_trajectories_data"):
            task["parent_trajectories"] = [
                StrategyTrajectory.from_dict(t) for t in task["parent_trajectories_data"]
            ]

        traj_data = _run_evolution_task(
            task=task,
            directions=directions,
            step_n=step_n,
            use_local=use_local,
            user_direction=user_direction,
            log_root=log_root,
            stop_event=None,
            quality_gate_cfg=quality_gate_cfg,
        )
        result_queue.put({
            "success": True,
            "task_idx": task_idx,
            "task": task,
            "traj_data": traj_data,
        })
    except Exception as e:
        import traceback
        result_queue.put({
            "success": False,
            "task_idx": task_idx,
            "task": task,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def _serialize_task_for_parallel(task: dict[str, Any]) -> dict[str, Any]:
    """Serialize task for use in child process (parent_trajectories are complex objects)."""
    serialized = task.copy()
    
    # RoundPhase -> string
    if "phase" in serialized and isinstance(serialized["phase"], RoundPhase):
        serialized["phase"] = serialized["phase"]
    
    # Convert parent_trajectories to serializable info
    if "parent_trajectories" in serialized:
        parent_trajs = serialized.get("parent_trajectories", [])
        serialized["parent_trajectory_ids"] = [p.trajectory_id for p in parent_trajs]
        serialized["parent_trajectories_data"] = [p.to_dict() for p in parent_trajs]
        serialized["parent_trajectories"] = []
    
    return serialized


def _run_tasks_parallel(
    tasks: list[dict[str, Any]],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    quality_gate_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run multiple evolution tasks in parallel.
    Returns list of results, each with task and traj_data.
    """
    if not tasks:
        return []
    
    result_queue = Queue()
    processes = []
    
    logger.info(f"Starting {len(tasks)} parallel evolution tasks")

    for idx, task in enumerate(tasks):
        serialized_task = _serialize_task_for_parallel(task)
        
        p = Process(
            target=_parallel_task_worker,
            args=(
                serialized_task,
                directions,
                step_n,
                use_local,
                user_direction,
                log_root,
                quality_gate_cfg,
                result_queue,
                idx,
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started task {idx}: phase={task['phase'].value}, direction={task['direction_id']}")

    results = []
    for _ in range(len(tasks)):
        result = result_queue.get()
        if result["success"]:
            original_task = tasks[result["task_idx"]]
            result["task"] = original_task
            result["traj_data"]["task"] = original_task
            results.append(result)
            logger.info(f"Task {result['task_idx']} completed")
        else:
            logger.error(f"Task {result['task_idx']} failed: {result['error']}")
            logger.error(result.get('traceback', ''))

    for p in processes:
        p.join()

    logger.info(f"Parallel tasks done: {len(results)}/{len(tasks)} succeeded")
    
    return results


def run_evolution_loop(
    initial_direction: str | None,
    evolution_cfg: dict[str, Any],
    exec_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
    stop_event: threading.Event | None = None,
    quality_gate_cfg: dict[str, Any] | None = None,
):
    """
    Run evolution loop: Original -> Mutation -> Crossover -> Mutation -> ...
    Supports parallel execution per phase.
    """
    quality_gate_cfg = quality_gate_cfg or {}
    from quantaalpha.core.conf import RD_AGENT_SETTINGS
    RD_AGENT_SETTINGS.use_file_lock = False
    logger.info("Evolution mode: file lock disabled to avoid deadlock")

    # Parse config
    num_directions = int(planning_cfg.get("num_directions", 5))
    max_rounds = int(evolution_cfg.get("max_rounds", 10))
    crossover_size = int(evolution_cfg.get("crossover_size", 2))
    crossover_n = int(evolution_cfg.get("crossover_n", 3))
    steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
    use_local = bool(exec_cfg.get("use_local", True))
    
    mutation_enabled = bool(evolution_cfg.get("mutation_enabled", True))
    crossover_enabled = bool(evolution_cfg.get("crossover_enabled", True))
    parent_selection_strategy = str(evolution_cfg.get("parent_selection_strategy", "best"))
    top_percent_threshold = float(evolution_cfg.get("top_percent_threshold", 0.3))
    crossover_intent_quota = dict(evolution_cfg.get("crossover_intent_quota") or {})
    crossover_candidate_filter = dict(evolution_cfg.get("crossover_candidate_filter") or {})
    log_root = str(logger.log_trace_path)
    parallel_enabled = bool(evolution_cfg.get("parallel_enabled", False))
    fresh_start = bool(evolution_cfg.get("fresh_start", True))
    cleanup_on_finish = bool(evolution_cfg.get("cleanup_on_finish", False))
    reselect_initial_factors = bool(evolution_cfg.get("reselect_initial_factors", False))
    max_empty_task_retries = max(1, int(evolution_cfg.get("max_empty_task_retries", 3)))
    empty_task_attempts: dict[tuple[str, int, int, str, tuple[str, ...]], int] = {}
    experiment_dir = Path(__file__).resolve().parents[2] / "experiment"
    active_data_domains = resolve_effective_factor_domains(
        os.getenv("FACTOR_DATA_DOMAINS"),
        data_mode=os.getenv("FACTOR_DATA_MODE"),
    )
    used_seed_identities = _new_seed_tracker()
    manual_initial_direction = str(initial_direction or "").strip() or None
    experiment_seed_direction = None
    if reselect_initial_factors and not manual_initial_direction:
        experiment_seed_direction = _build_domain_seed_direction(
            experiment_dir=experiment_dir,
            data_domains=active_data_domains,
            used_direction_identities=used_seed_identities,
        )
        if experiment_seed_direction:
            logger.info("Prepared one random seed direction for this evolution experiment.")
        else:
            logger.warning("reselect_initial_factors is enabled, but no usable experiment seed direction was found.")
    elif reselect_initial_factors:
        logger.info("Skipping random experiment seed because an explicit initial direction was provided.")

    # Generate initial directions
    planning_enabled = bool(planning_cfg.get("enabled", False))
    prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
    prompt_path = Path(__file__).parent / "prompts" / str(prompt_file)
    seeded_initial_direction = _merge_seed_direction(
        base_direction=manual_initial_direction,
        seed_direction=experiment_seed_direction,
        context_label="experiment",
    )
    
    if planning_enabled and seeded_initial_direction:
        directions = generate_parallel_directions(
            initial_direction=seeded_initial_direction,
            n=num_directions,
            prompt_file=prompt_path,
            max_attempts=int(planning_cfg.get("max_attempts", 5)),
            use_llm=bool(planning_cfg.get("use_llm", True)),
            allow_fallback=bool(planning_cfg.get("allow_fallback", True)),
            include_initial_direction=True,
        )
    elif planning_enabled:
        directions = [None] * num_directions
    else:
        directions = [seeded_initial_direction] if seeded_initial_direction else [None]

    directions = _dedupe_direction_list(directions)

    logger.info(f"Generated {len(directions)} exploration directions")
    for i, d in enumerate(directions):
        logger.info(f"  Direction {i}: {d}")

    pool_save_path = Path(log_root) / "trajectory_pool.json"
    mutation_prompt_path = Path(__file__).parent / "prompts" / "evolution_prompts.yaml"
    
    logger.info(f"Trajectory pool path: {pool_save_path} (fresh_start={fresh_start})")

    config = EvolutionConfig(
        num_directions=len(directions),
        steps_per_loop=steps_per_loop,
        max_rounds=max_rounds,
        mutation_enabled=mutation_enabled,
        crossover_enabled=crossover_enabled,
        crossover_size=crossover_size,
        crossover_n=crossover_n,
        prefer_diverse_crossover=True,
        parent_selection_strategy=parent_selection_strategy,
        top_percent_threshold=top_percent_threshold,
        crossover_intent_quota=crossover_intent_quota,
        crossover_candidate_filter=crossover_candidate_filter,
        parallel_enabled=parallel_enabled,
        pool_save_path=str(pool_save_path),
        mutation_prompt_path=str(mutation_prompt_path) if mutation_prompt_path.exists() else None,
        crossover_prompt_path=str(mutation_prompt_path) if mutation_prompt_path.exists() else None,
        fresh_start=fresh_start,
    )

    controller = EvolutionController(config)

    logger.info("="*60)
    logger.info("Starting evolution loop")
    logger.info(f"Config: directions={len(directions)}, max_rounds={max_rounds}, "
               f"crossover_size={crossover_size}, crossover_n={crossover_n}")
    logger.info(f"Phases: mutation={'on' if mutation_enabled else 'off'}, "
               f"crossover={'on' if crossover_enabled else 'off'}")
    if mutation_enabled and not crossover_enabled:
        logger.info("Mode: mutation only (Original -> Mutation -> ...)")
    elif crossover_enabled and not mutation_enabled:
        logger.info("Mode: crossover only (Original -> Crossover -> ...)")
    elif mutation_enabled and crossover_enabled:
        logger.info("Mode: full evolution (Original -> Mutation -> Crossover -> ...)")
    else:
        logger.info("Mode: original only (no evolution)")
    logger.info(
        f"Parent selection: {parent_selection_strategy}"
        + (
            f" (top_percent={top_percent_threshold})"
            if parent_selection_strategy == "top_percent_plus_random"
            else ""
        )
    )
    if parent_selection_strategy == "best":
        logger.info(
            "Crossover best-mode semantics: intent-driven parent pairing "
            f"(quota={crossover_intent_quota or 'default'}, filter={crossover_candidate_filter or 'default'})"
        )
    logger.info(f"Parallel execution: {'on' if parallel_enabled else 'off'}")
    logger.info("="*60)

    last_phase = controller._current_phase

    if parallel_enabled:
        while not controller.is_complete():
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending evolution loop")
                break

            current_phase = controller._current_phase
            if (
                reselect_initial_factors
                and last_phase == RoundPhase.CROSSOVER
                and current_phase != RoundPhase.CROSSOVER
            ):
                logger.info("Crossover phase ended. Reselecting new initial factors.")
                new_seed = _build_domain_seed_direction(
                    experiment_dir=experiment_dir,
                    data_domains=active_data_domains,
                    used_direction_identities=used_seed_identities,
                )
                if new_seed:
                    logger.info(f"New seed direction: {new_seed}")
                    if planning_enabled:
                        new_sub_directions = generate_parallel_directions(
                            initial_direction=new_seed,
                            n=num_directions,
                            prompt_file=prompt_path,
                            max_attempts=int(planning_cfg.get("max_attempts", 5)),
                            use_llm=bool(planning_cfg.get("use_llm", True)),
                            allow_fallback=bool(planning_cfg.get("allow_fallback", True)),
                            include_initial_direction=True,
                            excluded_directions=directions,
                        )
                    else:
                        new_sub_directions = [new_seed]

                    new_sub_directions = _dedupe_direction_list(new_sub_directions)
                    if new_sub_directions:
                        directions.extend(new_sub_directions)
                        controller.expand_directions(len(new_sub_directions))
                        logger.info(
                            f"Expanded directions with {len(new_sub_directions)} new tasks. Restarting Original phase."
                        )
                    else:
                        logger.info("No unique new directions were produced after crossover; skip expansion.")

            last_phase = controller._current_phase
            tasks = controller.get_all_tasks_for_current_phase()
            if not tasks:
                logger.info("Evolution complete: no more tasks")
                break

            current_phase = tasks[0]["phase"]
            current_round = tasks[0]["round_idx"]
            logger.info(f"Parallel phase: phase={current_phase.value}, round={current_round}, tasks={len(tasks)}")

            results = _run_tasks_parallel(
                tasks=tasks,
                directions=directions,
                step_n=steps_per_loop,
                use_local=use_local,
                user_direction=initial_direction,
                log_root=log_root,
                quality_gate_cfg={
                    **quality_gate_cfg,
                    "reselect_initial_factors": reselect_initial_factors,
                },
            )
            
            completed_tasks = []
            retry_pending = False
            for result in results:
                if result["success"]:
                    task = result["task"]
                    traj_data = result["traj_data"]
                    if _has_trajectory_output(traj_data):
                        trajectory = controller.create_trajectory_from_loop_result(
                            task=task,
                            hypothesis=traj_data.get("hypothesis"),
                            experiment=traj_data.get("experiment"),
                            feedback=traj_data.get("feedback"),
                        )
                        controller.report_task_complete(task, trajectory)
                        logger.info(
                            "Trajectory done: "
                            f"{trajectory.trajectory_id}, "
                            f"{_format_trajectory_metric_profile(trajectory)}"
                        )
                    else:
                        should_retry = _record_empty_task_attempt(
                            empty_task_attempts,
                            task,
                            max_empty_task_retries,
                        )
                        if should_retry:
                            retry_pending = True
                            logger.warning(
                                "Task did not finish a full factor/backtest/feedback loop; "
                                "will retry without counting it complete: "
                                f"phase={task['phase'].value}, round={task['round_idx']}, "
                                f"direction={task['direction_id']}"
                            )
                            continue
                        logger.warning(
                            "Empty task retry limit exhausted; skipping trajectory creation "
                            "and marking task attempted so the phase can continue: "
                            f"phase={task['phase'].value}, round={task['round_idx']}, "
                            f"direction={task['direction_id']}"
                        )
                    completed_tasks.append(task)

            if retry_pending:
                logger.info("Retrying phase because at least one task produced no usable trajectory output.")
            else:
                controller.advance_phase_after_parallel_completion(completed_tasks)

    else:
        last_phase = controller._current_phase
        while not controller.is_complete():
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending evolution loop")
                break
            
            current_phase = controller._current_phase
            if (
                reselect_initial_factors
                and last_phase == RoundPhase.CROSSOVER
                and current_phase != RoundPhase.CROSSOVER
            ):
                logger.info("Crossover phase ended. Reselecting new initial factors.")
                new_seed = _build_domain_seed_direction(
                    experiment_dir=experiment_dir,
                    data_domains=active_data_domains,
                    used_direction_identities=used_seed_identities,
                )
                if new_seed:
                    logger.info(f"New seed direction: {new_seed}")
                    if planning_enabled:
                        new_sub_directions = generate_parallel_directions(
                            initial_direction=new_seed,
                            n=num_directions,
                            prompt_file=prompt_path,
                            max_attempts=int(planning_cfg.get("max_attempts", 5)),
                            use_llm=bool(planning_cfg.get("use_llm", True)),
                            allow_fallback=bool(planning_cfg.get("allow_fallback", True)),
                            include_initial_direction=True,
                            excluded_directions=directions,
                        )
                    else:
                        new_sub_directions = [new_seed]

                    new_sub_directions = _dedupe_direction_list(new_sub_directions)
                    if new_sub_directions:
                        directions.extend(new_sub_directions)
                        controller.expand_directions(len(new_sub_directions))
                        logger.info(
                            f"Expanded directions with {len(new_sub_directions)} new tasks. Restarting Original phase."
                        )
                    else:
                        logger.info("No unique new directions were produced after crossover; skip expansion.")

            last_phase = controller._current_phase
            task = controller.get_next_task()
            if task is None:
                logger.info("Evolution complete: no more tasks")
                break

            logger.info(f"Running task: phase={task['phase'].value}, round={task['round_idx']}, direction={task['direction_id']}")

            try:
                traj_data = _run_evolution_task(
                    task=task,
                    directions=directions,
                    step_n=steps_per_loop,
                    use_local=use_local,
                    user_direction=initial_direction,
                    log_root=log_root,
                    stop_event=stop_event,
                    quality_gate_cfg={
                        **quality_gate_cfg,
                        "reselect_initial_factors": reselect_initial_factors,
                    },
                )
                if _has_trajectory_output(traj_data):
                    trajectory = controller.create_trajectory_from_loop_result(
                        task=task,
                        hypothesis=traj_data.get("hypothesis"),
                        experiment=traj_data.get("experiment"),
                        feedback=traj_data.get("feedback"),
                    )
                    controller.report_task_complete(task, trajectory)
                    logger.info(
                        "Task done: "
                        f"trajectory_id={trajectory.trajectory_id}, "
                        f"{_format_trajectory_metric_profile(trajectory)}"
                    )
                else:
                    should_retry = _record_empty_task_attempt(
                        empty_task_attempts,
                        task,
                        max_empty_task_retries,
                    )
                    if should_retry:
                        logger.warning(
                            "Task did not finish a full factor/backtest/feedback loop; "
                            "retrying without counting it complete: "
                            f"phase={task['phase'].value}, round={task['round_idx']}, "
                            f"direction={task['direction_id']}"
                        )
                        if task["phase"] == RoundPhase.MUTATION:
                            controller._mutation_idx = max(0, controller._mutation_idx - 1)
                        elif task["phase"] == RoundPhase.CROSSOVER:
                            controller._crossover_idx = max(0, controller._crossover_idx - 1)
                        continue
                    logger.warning(
                        "Empty task retry limit exhausted; skipping trajectory creation "
                        "and marking task attempted so the phase can continue: "
                        f"phase={task['phase'].value}, round={task['round_idx']}, "
                        f"direction={task['direction_id']}"
                    )
                    if task["phase"] == RoundPhase.ORIGINAL:
                        controller._directions_completed.add(task["direction_id"])
            except Exception as e:
                logger.error(f"Task failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    state_path = Path(log_root) / "evolution_state.json"
    controller.save_state(state_path)
    best_trajs = controller.get_best_trajectories(top_n=5)
    logger.info("="*60)
    logger.info(f"Evolution complete. Representative {len(best_trajs)} trajectories:")
    for i, t in enumerate(best_trajs):
        logger.info(
            f"  {i+1}. {t.trajectory_id}: "
            f"phase={t.phase.value}, {_format_trajectory_metric_profile(t)}"
        )
    logger.info(f"Pool stats: {controller.pool.get_statistics()}")
    logger.info("="*60)
    if cleanup_on_finish:
        logger.info("Cleaning up trajectory pool file...")
        controller.pool.cleanup_file()


@force_timeout()
def main(path=None, step_n=100, direction=None, stop_event=None, config_path=None, evolution_mode=None):
    """
    Autonomous alpha factor mining with optional evolution support.

    Args:
        path: Session path (for resume)
        step_n: Number of steps (default 100 = 20 loops * 5 steps/loop)
        direction: Initial direction
        stop_event: Stop event
        config_path: Run config file path
        evolution_mode: Enable evolution (None=from config, True/False=override)

    Evolution flow: Original -> Mutation -> Crossover -> Mutation -> ...

    You can continue running session by

    .. code-block:: python

        quantaalpha mine --direction "[Initial Direction]" --config_path configs/experiment.yaml

    """
    try:
        from quantaalpha.core.conf import RD_AGENT_SETTINGS
        logger.info("="*60)
        logger.info("Experiment config")
        logger.info(f"  Workspace: {RD_AGENT_SETTINGS.workspace_path}")
        logger.info(f"  Cache dir: {RD_AGENT_SETTINGS.pickle_cache_folder_path_str}")
        logger.info(f"  Cache namespace: {os.environ.get('PICKLE_CACHE_NAMESPACE', '<unset>')}")
        logger.info(f"  Cache enabled: {RD_AGENT_SETTINGS.cache_with_pickle}")
        logger.info(f"  Factor cache dir: {os.environ.get('FACTOR_CACHE_DIR', '<unset>')}")
        logger.info(f"  Prompt cache path: {os.environ.get('PROMPT_CACHE_PATH', '<unset>')}")
        logger.info(
            "  Knowledge load path: "
            f"{os.environ.get('FACTOR_CoSTEER_knowledge_base_path') or os.environ.get('FACTOR_COSTEER_KNOWLEDGE_BASE_PATH') or '<unset>'}"
        )
        logger.info(
            "  Knowledge persist path: "
            f"{os.environ.get('FACTOR_CoSTEER_new_knowledge_base_path') or os.environ.get('FACTOR_COSTEER_NEW_KNOWLEDGE_BASE_PATH') or '<unset>'}"
        )
        logger.info("="*60)

        # Config file default is controlled by .env.
        from quantaalpha.runtime import experiment_config_path

        config_default = experiment_config_path()
        config_file = Path(config_path) if config_path else config_default
        run_cfg = load_run_config(config_file)
        planning_cfg = (run_cfg.get("planning") or {}) if isinstance(run_cfg, dict) else {}
        exec_cfg = (run_cfg.get("execution") or {}) if isinstance(run_cfg, dict) else {}
        evolution_cfg = (run_cfg.get("evolution") or {}) if isinstance(run_cfg, dict) else {}
        quality_gate_cfg = (run_cfg.get("quality_gate") or {}) if isinstance(run_cfg, dict) else {}
        factor_cfg = (run_cfg.get("factor") or {}) if isinstance(run_cfg, dict) else {}
        backtest_cfg = (run_cfg.get("backtest") or {}) if isinstance(run_cfg, dict) else {}

        # Frequency / prompt / backtest mode switches
        data_mode, prompt_mode, backtest_mode, data_domains = resolve_factor_modes(factor_cfg)
        backtest_engine = resolve_backtest_engine(backtest_cfg)
        tq_config_path = backtest_cfg.get("tq_upstream_config")
        factors_per_hypothesis = max(1, int(
            os.environ.get("FACTOR_FACTORS_PER_HYPOTHESIS")
            or factor_cfg.get("factors_per_hypothesis")
            or 3
        ))
        if tq_config_path:
            tq_path = Path(tq_config_path)
            if tq_path.is_absolute():
                tq_config_path = str(tq_path)
            elif (config_file.parent.parent / tq_path).exists():
                tq_config_path = str((config_file.parent.parent / tq_path).resolve())
            else:
                tq_config_path = str((config_file.parent / tq_path).resolve())
        configure_backtest_engine(backtest_engine, tq_config_path)
        effective_backtest_cfg = load_backtest_config(tq_config_path)
        backtest_universe = str(
            ((effective_backtest_cfg.get("spec_defaults") or {}).get("universe") or "standards")
        ).strip() or "standards"
        transform_spec = effective_backtest_cfg.get("transform_spec") or {}

        os.environ["FACTOR_DATA_MODE"] = data_mode
        os.environ["FACTOR_PROMPT_MODE"] = prompt_mode
        os.environ["FACTOR_BACKTEST_MODE"] = backtest_mode
        os.environ["FACTOR_DATA_DOMAINS"] = "/".join(data_domains)
        os.environ["FACTOR_FACTORS_PER_HYPOTHESIS"] = str(factors_per_hypothesis)

        logger.info("="*60)
        logger.info("Factor mining mode switches")
        logger.info(f"  data_mode: {data_mode}")
        logger.info(f"  prompt_mode: {prompt_mode}")
        logger.info(f"  backtest_mode: {backtest_mode}")
        logger.info(f"  factors_per_hypothesis: {factors_per_hypothesis}")
        logger.info(f"  data_domains: {', '.join(data_domains)}")
        logger.info(f"  backtest_universe: {backtest_universe}")
        logger.info(f"  backtest_transform_spec: {transform_spec}")
        logger.info(f"  backtest_engine: {backtest_engine}")
        logger.info(f"  data_domain_summary: {describe_factor_domains(data_domains)}")
        logger.info("="*60)

        if evolution_mode is not None:
            use_evolution = evolution_mode
        else:
            use_evolution = bool(evolution_cfg.get("enabled", False))

        if step_n is None or step_n == 100:
            if exec_cfg.get("step_n") is not None:
                step_n = exec_cfg.get("step_n")
            else:
                max_loops = int(exec_cfg.get("max_loops", 10))
                steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
                step_n = max_loops * steps_per_loop

        use_local = os.getenv("USE_LOCAL", "True").lower()
        use_local = True if use_local in ["true", "1"] else False
        if exec_cfg.get("use_local") is not None:
            use_local = bool(exec_cfg.get("use_local"))
        exec_cfg["use_local"] = use_local
        
        logger.info(f"Use {'Local' if use_local else 'Docker container'} to execute factor backtest")
        
        if use_evolution and path is None:
            logger.info("="*60)
            logger.info("Evolution mode: Original -> Mutation -> Crossover loop")
            logger.info("="*60)
            
            run_evolution_loop(
                initial_direction=direction,
                evolution_cfg=evolution_cfg,
                exec_cfg=exec_cfg,
                planning_cfg=planning_cfg,
                stop_event=stop_event,
                quality_gate_cfg=quality_gate_cfg,
            )
        
        elif path is None:
            planning_enabled = bool(planning_cfg.get("enabled", False))
            n_dirs = int(planning_cfg.get("num_directions", 1))
            max_attempts = int(planning_cfg.get("max_attempts", 5))
            use_llm = bool(planning_cfg.get("use_llm", True))
            allow_fallback = bool(planning_cfg.get("allow_fallback", True))
            prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
            prompt_path = Path(__file__).parent / "prompts" / str(prompt_file)
            if planning_enabled and direction:
                directions = generate_parallel_directions(
                    initial_direction=direction,
                    n=n_dirs,
                    prompt_file=prompt_path,
                    max_attempts=max_attempts,
                    use_llm=use_llm,
                    allow_fallback=allow_fallback,
                    include_initial_direction=True,
                )
            else:
                directions = [direction] if direction else [None]

            if planning_enabled:
                directions = _dedupe_direction_list(directions)

            log_root = exec_cfg.get("branch_log_root") or "log"
            log_prefix = exec_cfg.get("branch_log_prefix") or "branch"
            use_branch_logs = planning_enabled and len(directions) > 1
            parallel_execution = bool(exec_cfg.get("parallel_execution", False))

            if parallel_execution and len(directions) > 1:
                procs: list[Process] = []
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}")
                    p = Process(
                        target=_run_branch,
                        args=(dir_text, step_n, use_local, idx, log_root if use_branch_logs else "", log_prefix),
                    )
                    p.start()
                    procs.append(p)
                for p in procs:
                    p.join()
            else:
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}")
                    model_loop = AlphaAgentLoop(
                        ALPHA_AGENT_FACTOR_PROP_SETTING,
                        potential_direction=dir_text,
                        stop_event=stop_event,
                        use_local=use_local,
                        quality_gate_config=quality_gate_cfg,
                    )
                    model_loop.user_initial_direction = direction
                    model_loop.run(step_n=step_n, stop_event=stop_event)
        else:
            model_loop = AlphaAgentLoop.load(path, use_local=use_local)
            model_loop.run(step_n=step_n, stop_event=stop_event)
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise
    finally:
        logger.info("Run finished or terminated")

if __name__ == "__main__":
    fire.Fire(main)










