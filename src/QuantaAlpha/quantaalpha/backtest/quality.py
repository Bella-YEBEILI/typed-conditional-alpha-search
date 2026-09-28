from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .performance_engine import DEFAULT_PARAMS

STYLE_FIELDS = ("Beta", "Liq", "Mom", "Nlsize", "Rev", "Size", "Vol")


class FactorPlotter:
    """Manual research plots for standalone TQ factor backtests."""

    def plot_result(
        self,
        factor_id: str,
        factor_result: dict[str, Any],
        params: dict[str, Any] = DEFAULT_PARAMS,
        output_path: str | Path | None = None,
        show: bool = False,
    ):
        start = params["start"]
        end = params["end"]
        cost = params["cost"]

        long_rets = factor_result.get("long_rets", pd.Series(dtype=float)).loc[start:end]
        long_turnovers = factor_result.get("long_turnovers", pd.Series(dtype=float)).loc[start:end]
        short_rets = factor_result.get("short_rets", pd.Series(dtype=float)).loc[start:end]
        short_turnovers = factor_result.get("short_turnovers", pd.Series(dtype=float)).loc[start:end]
        rankics = factor_result.get("rankics", pd.Series(dtype=float)).loc[start:end]
        long_nums = factor_result.get("long_nums", pd.Series(dtype=float)).loc[start:end]
        coverages = factor_result.get("coverages", pd.Series(dtype=float)).loc[start:end]

        long_netrets = long_rets - (long_turnovers / 2 * cost)
        short_netrets = short_rets - (short_turnovers / 2 * cost)
        ls_rets = (long_rets + short_rets) / 2
        ls_netrets = (long_netrets + short_netrets) / 2
        ls_turnovers = (long_turnovers + short_turnovers) / 2

        group_rets = factor_result.get("group_rets", pd.DataFrame()).loc[start:end]
        demeaned_group_rets = group_rets.sub(group_rets.mean(axis=1), axis=0)

        fig, axes = plt.subplots(3, 2, figsize=(10, 7), constrained_layout=True)
        fig.suptitle(factor_id, fontsize=12)

        nav_df = pd.DataFrame(
            {
                "long_rets": (1 + long_rets.fillna(0)).cumprod(),
                "long_netrets": (1 + long_netrets.fillna(0)).cumprod(),
                "ls_rets": (1 + ls_rets.fillna(0)).cumprod(),
                "ls_netrets": (1 + ls_netrets.fillna(0)).cumprod(),
            }
        )
        nav_df.plot(ax=axes[0, 0], title="Return NAV", legend=True, linewidth=1)
        axes[0, 0].legend(fontsize=8, loc="upper left")

        pd.DataFrame({"rankics": rankics.fillna(0).cumsum()}).plot(
            ax=axes[0, 1], title="RankIC CumSum", legend=True, linewidth=1
        )
        axes[0, 1].legend(fontsize=8, loc="upper left")

        group_rets.fillna(0).cumsum().plot(ax=axes[1, 0], title="Group CumRet", legend=False, linewidth=0.8)
        demeaned_group_rets.fillna(0).cumsum().plot(
            ax=axes[1, 1], title="Demeaned Group CumRet", legend=False, linewidth=0.8
        )
        handles, labels = axes[1, 0].get_legend_handles_labels()
        if handles:
            axes[1, 0].legend(handles, [x.replace("group_", "") for x in labels], fontsize=7, loc="upper left", ncol=2)
        handles, labels = axes[1, 1].get_legend_handles_labels()
        if handles:
            axes[1, 1].legend(handles, [x.replace("group_", "") for x in labels], fontsize=7, loc="upper left", ncol=2)

        pd.DataFrame({"long_turnovers": long_turnovers, "ls_turnovers": ls_turnovers}).plot(
            ax=axes[2, 0], title="Turnover", legend=True, linewidth=1
        )
        handles, labels = axes[2, 0].get_legend_handles_labels()
        axes[2, 0].legend(handles, [x.replace("_turnovers", "") for x in labels], fontsize=8, loc="upper left")

        ax_left = axes[2, 1]
        ax_right = ax_left.twinx()
        long_nums.plot(ax=ax_left, color="tab:blue", label="long_nums", title="Long Nums / Coverage", linewidth=1)
        coverages.plot(ax=ax_right, color="tab:orange", label="coverages", linewidth=1)
        h1, l1 = ax_left.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax_left.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        if ax_right.get_legend() is not None:
            ax_right.get_legend().remove()

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.title.set_fontsize(10)

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=160, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig


class FactorQualityAnalyzer:
    """Manual quality diagnostics such as decile summaries and Barra-style exposures."""

    def __init__(self, data_provider):
        self.dp = data_provider

    @staticmethod
    def _rowwise_corr(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
        values = []
        for idx in left.index:
            x = left.loc[idx]
            y = right.loc[idx]
            mask = x.notna() & y.notna()
            if mask.sum() < 2:
                values.append(np.nan)
                continue
            values.append(float(x[mask].corr(y[mask])))
        return pd.Series(values, index=left.index, dtype=float)

    def _mask_factor(self, factor_value: pd.DataFrame, universe_name: str = "standards") -> pd.DataFrame:
        universe = self.dp.get_single_data(universe_name).astype(bool)
        factor_value, universe = factor_value.align(universe, join="inner", axis=0)
        return factor_value.where(universe)

    def compute_style_exposures(
        self,
        factor_value: pd.DataFrame,
        style_fields: tuple[str, ...] = STYLE_FIELDS,
        universe_name: str = "standards",
    ) -> pd.DataFrame:
        masked_factor = self._mask_factor(factor_value, universe_name=universe_name)
        rows: dict[str, pd.Series] = {}
        for field in style_fields:
            try:
                style_df = self.dp.get_single_data(field)
            except KeyError:
                continue
            style_df = style_df.reindex(index=masked_factor.index, columns=masked_factor.columns)
            rows[field] = self._rowwise_corr(masked_factor, style_df)
        return pd.DataFrame(rows)

    def summarize_style_exposures(self, factor_value: pd.DataFrame, universe_name: str = "standards") -> pd.DataFrame:
        exposures = self.compute_style_exposures(factor_value, universe_name=universe_name)
        if exposures.empty:
            return pd.DataFrame(columns=["mean_corr", "mean_abs_corr", "last_corr"])
        summary = pd.DataFrame(
            {
                "mean_corr": exposures.mean(axis=0),
                "mean_abs_corr": exposures.abs().mean(axis=0),
                "last_corr": exposures.iloc[-1],
            }
        )
        summary.index.name = "style_factor"
        return summary.sort_values("mean_abs_corr", ascending=False)

    @staticmethod
    def summarize_group_returns(factor_result: dict[str, Any]) -> pd.DataFrame:
        group_rets = factor_result.get("group_rets", pd.DataFrame())
        if not isinstance(group_rets, pd.DataFrame) or group_rets.empty:
            return pd.DataFrame(columns=["mean_daily_return", "final_cum_return"])

        summary = pd.DataFrame(
            {
                "mean_daily_return": group_rets.mean(axis=0),
                "final_cum_return": (1 + group_rets.fillna(0)).cumprod().iloc[-1] - 1,
            }
        )
        summary.index.name = "group"
        return summary.sort_index()

    @staticmethod
    def summarize_factor_result(factor_result: dict[str, Any]) -> dict[str, float | None]:
        long_rets = factor_result.get("long_rets", pd.Series(dtype=float))
        rankics = factor_result.get("rankics", pd.Series(dtype=float))
        coverages = factor_result.get("coverages", pd.Series(dtype=float))
        long_turnovers = factor_result.get("long_turnovers", pd.Series(dtype=float))
        return {
            "avg_daily_long_ret": None if long_rets.empty else float(long_rets.mean()),
            "avg_rankic": None if rankics.empty else float(rankics.mean()),
            "avg_coverage": None if coverages.empty else float(coverages.mean()),
            "avg_long_turnover": None if long_turnovers.empty else float(long_turnovers.mean()),
        }

    def build_report(
        self,
        factor_name: str,
        factor_value: pd.DataFrame,
        factor_result: dict[str, Any] | None = None,
        universe_name: str = "standards",
    ) -> dict[str, Any]:
        factor_result = factor_result or {}
        style_summary = self.summarize_style_exposures(factor_value, universe_name=universe_name)
        group_summary = self.summarize_group_returns(factor_result)
        return {
            "factor_name": factor_name,
            "result_summary": self.summarize_factor_result(factor_result),
            "group_summary": group_summary.to_dict(orient="index"),
            "style_exposures": style_summary.to_dict(orient="index"),
        }
