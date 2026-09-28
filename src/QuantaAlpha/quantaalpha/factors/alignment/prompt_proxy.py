from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

from quantaalpha.core.prompts import Prompts
from quantaalpha.factors import data_domains as factor_data_domains
from quantaalpha.factors.combined_domain_contract import is_joint_pv_minutes_run


def resolve_prompt_mode() -> str:
    mode = os.getenv("FACTOR_PROMPT_MODE") or os.getenv("FACTOR_DATA_MODE") or "daily"
    return factor_data_domains.normalize_factor_mode(mode) or "daily"


class DomainPromptProxy:
    def __init__(self, base_file: str, prompt_dir: Path | str):
        self._base_file = base_file
        self._prompt_dir = Path(prompt_dir)
        self._cache: dict[tuple[str, tuple[str, ...]], dict] = {}

    def _domain_prompt_files(self, domains: tuple[str, ...]) -> list[Path]:
        base_path = Path(self._base_file)
        stem = base_path.stem
        suffix = base_path.suffix or ".yaml"
        effective_domains = list(domains)
        if is_joint_pv_minutes_run(domains):
            effective_domains.append("joint")
        files: list[Path] = []
        for domain in effective_domains:
            candidate = self._prompt_dir / f"{stem}_domain_{domain}{suffix}"
            if candidate.exists():
                files.append(candidate)
        return files

    def _append_prompt_values(self, base, extra):
        if isinstance(base, dict) and isinstance(extra, dict):
            merged = deepcopy(base)
            for key, value in extra.items():
                if key in merged:
                    merged[key] = self._append_prompt_values(merged[key], value)
                else:
                    merged[key] = deepcopy(value)
            return merged
        if isinstance(base, str) and isinstance(extra, str):
            extra_text = extra.strip()
            if not extra_text:
                return base
            base_text = base.rstrip()
            return f"{base_text}\n\n{extra_text}" if base_text else extra_text
        return deepcopy(extra)

    def _load(self, mode: str, domains: tuple[str, ...]) -> dict:
        cache_key = (mode, domains)
        if cache_key in self._cache:
            return self._cache[cache_key]

        base = dict(Prompts(file_path=self._prompt_dir / self._base_file))
        for extra_path in self._domain_prompt_files(domains):
            extra = dict(Prompts(file_path=extra_path))
            base = self._append_prompt_values(base, extra)
        self._cache[cache_key] = base
        return base

    def __getitem__(self, key):
        mode = resolve_prompt_mode()
        domains = factor_data_domains.resolve_factor_domains()
        return self._load(mode, domains)[key]

    def get(self, key, default=None):
        mode = resolve_prompt_mode()
        domains = factor_data_domains.resolve_factor_domains()
        return self._load(mode, domains).get(key, default)
