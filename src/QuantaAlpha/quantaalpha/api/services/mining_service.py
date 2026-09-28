from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from quantaalpha.runtime import project_root, experiment_config_path

logger = logging.getLogger(__name__)

DIRECTION_FILE_MAP = {
    "pv": "original_direction_daily.json",
    "minutes": "original_direction_minutes.json",
    "joint": "original_direction_joint.json",
    "fundamental": "original_direction_cross_sectional.json",
}

MODE_TO_DATA_MODE = {
    "pv": "daily",
    "minutes": "minutes",
    "joint": "daily",
    "fundamental": "fundamental",
}

MODE_TO_DOMAINS = {
    "pv": "pv",
    "minutes": "minutes",
    "joint": "pv/minutes",
    "fundamental": "fundamental",
}


def _normalize_direction_desc(desc: str) -> str:
    import re

    if not desc:
        return desc

    # Type B: "Portfolio N, K factors: NAME, fields=[...], expr=EXPR | English desc || ..."
    # Strip English prose after " | " within each factor segment
    segments = desc.split(" || ")
    cleaned = []
    for seg in segments:
        pipe_pos = seg.find(" | ")
        if pipe_pos > 0:
            before = seg[:pipe_pos].strip()
            after = seg[pipe_pos + 3:].strip()
            if re.match(r"[A-Z]", after) and not re.match(r"(fields|expr|inputs)=", after):
                seg = before
        cleaned.append(seg.strip())
    result = " || ".join(cleaned)

    # Type A: very long GA descriptions — extract factor names and truncate
    if len(result) > 300:
        m = re.match(r"(Portfolio \d+)[^:]*:\s*", result)
        prefix = m.group(1) if m else ""
        names = re.findall(r"(\w+) from \w+\.py", result)
        if names and prefix:
            result = f"{prefix}, {len(names)} factors: {', '.join(names)}"
        else:
            result = result[:297] + "..."

    return result


class MiningService:
    def get_directions(self, mode: str) -> dict[str, Any]:
        filename = DIRECTION_FILE_MAP.get(mode)
        if not filename:
            return {"mode": mode, "source_file": "", "portfolios": []}

        path = project_root() / "experiment" / filename
        if not path.exists():
            return {"mode": mode, "source_file": filename, "portfolios": []}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"mode": mode, "source_file": filename, "portfolios": []}

        portfolios = []
        for i, entry in enumerate(data.get("factor_portfolios", [])):
            desc = entry.get("description", "")
            if not desc and isinstance(entry, dict):
                desc = json.dumps(entry, ensure_ascii=False)[:200]
            portfolios.append({"portfolio_id": i, "description": _normalize_direction_desc(desc)})

        return {
            "mode": mode,
            "source_file": filename,
            "portfolios": portfolios,
        }

    def run_mining_task(
        self,
        mode: str,
        directions: list[str] | None,
        custom_direction: str | None,
        config: dict[str, Any],
        stop_event: threading.Event,
        library_suffix: str = "",
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        direction_text = custom_direction or ""
        if directions:
            direction_text = "\n".join(directions)

        data_mode = MODE_TO_DATA_MODE.get(mode, "daily")
        data_domains = MODE_TO_DOMAINS.get(mode, "pv")

        os.environ["FACTOR_DATA_MODE"] = data_mode
        os.environ["FACTOR_DATA_DOMAINS"] = data_domains

        # 设置因子库后缀，使挖掘出的因子存入独立的库文件
        if library_suffix:
            os.environ["FACTOR_LIBRARY_SUFFIX"] = library_suffix

        config_file = experiment_config_path()

        try:
            from quantaalpha.pipeline.factor_mining import load_run_config, run_evolution_loop

            run_cfg = load_run_config(config_file)

            planning_cfg = run_cfg.get("planning") or {}
            exec_cfg = run_cfg.get("execution") or {}
            evolution_cfg = run_cfg.get("evolution") or {}
            quality_gate_cfg = run_cfg.get("quality_gate") or {}

            evolution_cfg["max_rounds"] = config.get("max_rounds", evolution_cfg.get("max_rounds", 11))
            evolution_cfg["mutation_enabled"] = config.get("mutation_enabled", True)
            evolution_cfg["crossover_enabled"] = config.get("crossover_enabled", True)
            evolution_cfg["parallel_enabled"] = config.get("parallel_enabled", True)
            planning_cfg["num_directions"] = config.get("num_directions", planning_cfg.get("num_directions", 5))

            factor_cfg = run_cfg.get("factor") or {}
            factor_cfg["factors_per_hypothesis"] = config.get(
                "factors_per_hypothesis", factor_cfg.get("factors_per_hypothesis", 3)
            )

            os.environ["FACTOR_FACTORS_PER_HYPOTHESIS"] = str(
                max(1, int(factor_cfg["factors_per_hypothesis"]))
            )

            logger.info(f"启动挖掘任务: mode={mode}, direction={direction_text[:100]}, "
                         f"factors_per_hypothesis={os.environ['FACTOR_FACTORS_PER_HYPOTHESIS']}")

            run_evolution_loop(
                initial_direction=direction_text or None,
                evolution_cfg=evolution_cfg,
                exec_cfg=exec_cfg,
                planning_cfg=planning_cfg,
                stop_event=stop_event,
                quality_gate_cfg=quality_gate_cfg,
            )

            return {"status": "completed"}

        except Exception as e:
            logger.error(f"挖掘任务失败: {e}", exc_info=True)
            raise
