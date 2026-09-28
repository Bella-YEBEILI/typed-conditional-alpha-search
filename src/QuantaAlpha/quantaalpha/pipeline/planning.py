from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from quantaalpha.log import logger
from quantaalpha.llm.client import APIBackend, robust_json_parse
from quantaalpha.runtime import expand_env_placeholders


def _load_prompts(prompt_file: Path) -> dict[str, str]:
    if not prompt_file.exists():
        return {}
    try:
        return yaml.safe_load(prompt_file.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Failed to load planning prompts: {exc}")
        return {}


def _parse_directions(message: str, n: int) -> list[str] | None:
    try:
        data = robust_json_parse(message)
    except Exception:
        return None
    arr = data.get("directions") if isinstance(data, dict) else None
    if not isinstance(arr, list):
        return None
    vals = [str(x).strip() for x in arr if isinstance(x, str) and x.strip()]
    return vals if len(vals) >= n else None


def _normalize_direction_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _dedupe_directions(candidates: list[str], n: int, excluded: list[str] | None = None) -> list[str]:
    seen = {_normalize_direction_text(text) for text in (excluded or []) if str(text or "").strip()}
    unique: list[str] = []
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        normalized = _normalize_direction_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(text)
        if len(unique) >= n:
            break
    return unique


def _fallback_directions(initial_direction: str, n: int) -> list[str]:
    base = initial_direction.strip() if initial_direction else "market microstructure"
    patterns = [
        f"{base} + orthogonal cross-sectional ranking logic under the same active domain constraints",
        f"{base} + alternative temporal aggregation under the same field semantics",
        f"{base} + participation and liquidity adjustment without changing data domains",
        f"{base} + regime-conditioned normalization with the same allowed inputs",
        f"{base} + robustness-oriented simplification of the core economic mechanism",
        f"{base} + lower-correlation confirmation signal within the same domain",
        f"{base} + alternative scaling and neutralization design under the same constraints",
        f"{base} + stability-focused variant using a different construction path",
    ]
    out = []
    for i in range(n):
        out.append(patterns[i % len(patterns)])
    return out


def generate_parallel_directions(
    initial_direction: str,
    n: int,
    prompt_file: Path,
    max_attempts: int = 5,
    use_llm: bool = True,
    allow_fallback: bool = True,
    include_initial_direction: bool = False,
    excluded_directions: list[str] | None = None,
) -> list[str]:
    n = max(1, int(n))
    initial_text = str(initial_direction or "").strip()
    excluded = list(excluded_directions or [])
    collected: list[str] = [initial_text] if include_initial_direction and initial_text else []
    if len(_dedupe_directions(collected, n, excluded)) >= n:
        return _dedupe_directions(collected, n, excluded)

    prompts = _load_prompts(prompt_file)
    sys_tpl = prompts.get("system", "")
    user_tpl = prompts.get("user", "")
    output_format = prompts.get("output_format", "")
    request_n = max(n, n + 4)

    system_prompt = sys_tpl.format(initial_direction=initial_direction, n=request_n)
    user_prompt = user_tpl.format(initial_direction=initial_direction, n=request_n)
    if output_format:
        if "{n}" in output_format:
            output_format = output_format.replace("{n}", str(request_n))
        user_prompt = f"{user_prompt}\n\n{output_format}"

    if not use_llm:
        fallback = _fallback_directions(initial_direction, request_n) if allow_fallback else []
        return _dedupe_directions(collected + fallback, n, excluded)

    for attempt in range(1, max_attempts + 1):
        try:
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=False
            )
            directions = _parse_directions(resp, request_n)
            if directions:
                merged = _dedupe_directions(collected + directions, n, excluded)
                if len(merged) >= n:
                    return merged[:n]
            system_prompt += "\n\nStrictly output valid JSON. No extra text."
            logger.warning(f"Planning parse failed (attempt {attempt}), retrying...")
        except Exception as exc:
            logger.warning(f"Planning LLM call failed (attempt {attempt}): {exc}")

    fallback = _fallback_directions(initial_direction, request_n) if allow_fallback else []
    return _dedupe_directions(collected + fallback, n, excluded)


def load_run_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        return expand_env_placeholders(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    except Exception as exc:
        logger.warning(f"Failed to load run config: {exc}")
        return {}

