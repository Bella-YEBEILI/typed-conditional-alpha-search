from __future__ import annotations

import os
from pathlib import Path

from quantaalpha.backtest import FilesystemDataProvider, load_backtest_config
from quantaalpha.factors.alignment.registry import get_allowed_field_names
from quantaalpha.factors.data_domains import describe_factor_domains, resolve_factor_domains
from quantaalpha.runtime import data_root

DEFAULT_SOURCE_DATA_DIR = data_root()


def _resolve_source_data_dir() -> Path:
    root = os.getenv('QUANTAALPHA_DATA_ROOT')
    if root:
        return Path(root)

    env_data_dir = os.getenv('QUANTAALPHA_SOURCE_DATA_DIR')
    if env_data_dir:
        return Path(env_data_dir)

    try:
        config = load_backtest_config(os.getenv('TQ_UPSTREAM_CONFIG_PATH'))
        cfg_value = (config.get('tq') or {}).get('data_dir')
        if cfg_value:
            return Path(cfg_value)
    except Exception:
        pass

    return DEFAULT_SOURCE_DATA_DIR


def get_data_folder_intro(
    fname_reg: str = '.*',
    flags=0,
    variable_mapping=None,
    use_local: bool = True,
) -> str:
    del fname_reg, flags, variable_mapping, use_local

    domains = resolve_factor_domains()
    data_dir = _resolve_source_data_dir()
    lines = [
        'TQ upstream source data overview',
        f'- root: {data_dir}',
        '- mode: filesystem provider over TQ-compatible source data',
        f'- {describe_factor_domains(domains)}',
    ]

    if data_dir.exists():
        provider = FilesystemDataProvider(data_dir)
        available = set(provider.list_datas())
        allowed = sorted(available.intersection(get_allowed_field_names(domains)))
        preview = ', '.join(allowed[:48])
        if len(allowed) > 48:
            preview = f'{preview}, ...'
        lines.append(f"- available runtime fields: {preview or 'none'}")
    else:
        fallback_fields = ', '.join(sorted(get_allowed_field_names(domains)))
        lines.append(f'- data directory not found locally; configured allowlist: {fallback_fields}')

    lines.append('- note: factor mining/backtest should use the TQ upstream runtime and not legacy factor-template data/backtest path.')
    return '\n'.join(lines)

