"""Backward-compatible shim for the backtest workflow entrypoint."""

import fire

from quantaalpha.backtest.workflow import main

__all__ = ["main"]


if __name__ == "__main__":
    fire.Fire(main)
