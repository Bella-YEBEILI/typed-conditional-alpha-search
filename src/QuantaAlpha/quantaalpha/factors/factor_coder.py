"""Neutral entry points for factor coder components used by the main mining flow."""

from __future__ import annotations

from quantaalpha.factors.coder import FactorCoSTEER as _FullCodeFactorCoSTEER
from quantaalpha.factors.coder import FactorCoder as _ExpressionFactorCoder
from quantaalpha.factors.data_domains import resolve_factor_domains


def _should_use_full_code_coder() -> bool:
    return "minutes" in resolve_factor_domains()


class FactorCoder:
    """Route minute-domain runs to the full-code coder and keep expressions for daily flows."""

    def __new__(cls, scen, *args, **kwargs):
        backend_cls = _FullCodeFactorCoSTEER if _should_use_full_code_coder() else _ExpressionFactorCoder
        return backend_cls(scen, *args, **kwargs)


class FactorParser(FactorCoder):
    """Keep legacy parser imports aligned with the active runtime coder."""


FactorCoSTEER = _FullCodeFactorCoSTEER

__all__ = ["FactorCoSTEER", "FactorParser", "FactorCoder"]
