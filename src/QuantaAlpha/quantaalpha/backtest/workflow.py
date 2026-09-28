"""Shared backtest workflow entrypoints kept inside the backtest package."""

from __future__ import annotations

from pathlib import Path

import fire

from quantaalpha.backtest import configure_backtest_engine, resolve_backtest_engine
from quantaalpha.pipeline.loop import BacktestLoop
from quantaalpha.pipeline.planning import load_run_config
from quantaalpha.pipeline.settings import FACTOR_BACK_TEST_PROP_SETTING
from quantaalpha.runtime import experiment_config_path


def main(path=None, step_n=None, factor_path=None):
    """Run the session-based backtest loop used by the main CLI."""
    config_file = experiment_config_path()
    project_root = config_file.parent.parent
    run_cfg = load_run_config(config_file)
    backtest_cfg = (run_cfg.get("backtest") or {}) if isinstance(run_cfg, dict) else {}
    backtest_engine = resolve_backtest_engine(backtest_cfg)
    tq_config_path = backtest_cfg.get("tq_upstream_config")
    if tq_config_path:
        tq_path = Path(tq_config_path)
        if tq_path.is_absolute():
            tq_config_path = str(tq_path)
        elif (project_root / tq_path).exists():
            tq_config_path = str((project_root / tq_path).resolve())
        else:
            tq_config_path = str((config_file.parent / tq_path).resolve())
    configure_backtest_engine(backtest_engine, tq_config_path)

    if path is None:
        model_loop = BacktestLoop(FACTOR_BACK_TEST_PROP_SETTING, factor_path=factor_path)
    else:
        model_loop = BacktestLoop.load(path)
    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
