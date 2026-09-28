from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantaalpha.backtest.analysis import div
from quantaalpha.backtest.numbafunc import cs_corr_2d, cs_mean_1d, cs_mean_2d, cs_rank_2d

from .profiles import DEFAULT_PROFILE_ID, PROFILES


class FactorResultEngine:
    def __init__(self, data_provider):
        self.dp = data_provider
        self.data: dict[str, pd.DataFrame] = {}

    def _get_data(self, field_name: str):
        data = self.data.get(field_name)
        if data is None:
            data = self.dp.get_single_data(field_name)
            self.data[field_name] = data
        return data

    def _get_adj_opens(self):
        adj_opens = self.data.get("adj_opens")
        if adj_opens is None:
            adj_opens = self._get_data("opens") * self._get_data("adj_factors")
            self.data["adj_opens"] = adj_opens
        return adj_opens

    def _get_adj_closes(self):
        adj_closes = self.data.get("adj_closes")
        if adj_closes is None:
            adj_closes = self._get_data("closes") * self._get_data("adj_factors")
            self.data["adj_closes"] = adj_closes
        return adj_closes

    def _get_o2c(self):
        o2c = self.data.get("o2c")
        if o2c is None:
            o2c = div(self._get_adj_closes(), self._get_adj_opens()) - 1
            self.data["o2c"] = o2c
        return o2c

    def _get_o2o(self):
        o2o = self.data.get("o2o")
        if o2o is None:
            adj_opens = self._get_adj_opens()
            o2o = div(adj_opens, adj_opens.shift(1)) - 1
            self.data["o2o"] = o2o
        return o2o

    def _get_c2o(self):
        c2o = self.data.get("c2o")
        if c2o is None:
            c2o = div(self._get_adj_opens(), self._get_adj_closes().shift(1)) - 1
            self.data["c2o"] = c2o
        return c2o

    def _get_c2c(self):
        c2c = self.data.get("c2c")
        if c2c is None:
            c2c = div(self._get_adj_closes(), self._get_adj_closes().shift(1)) - 1
            self.data["c2c"] = c2c
        return c2c

    def _resolve_tradables(self, dates: pd.Index, stocks: pd.Index, apply_suspend_mask: bool) -> np.ndarray:
        if apply_suspend_mask:
            return self._get_data("tradables").reindex(index=dates, columns=stocks).fillna(False).to_numpy(
                dtype=bool,
                copy=False,
            )
        return np.ones((len(dates), len(stocks)), dtype=bool)

    def _resolve_limit_masks(
        self,
        dates: pd.Index,
        stocks: pd.Index,
        mode: int,
        apply_limit_mask: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not apply_limit_mask:
            zeros = np.zeros((len(dates), len(stocks)), dtype=bool)
            return zeros, zeros
        if mode == 1:
            buy_field = "limit_up_cto"
            sell_field = "limit_down_cto"
        else:
            buy_field = "limit_up_ctc"
            sell_field = "limit_down_ctc"
        buylimit = self._get_data(buy_field).reindex(index=dates, columns=stocks).fillna(True).to_numpy(
            dtype=bool,
            copy=False,
        )
        selllimit = self._get_data(sell_field).reindex(index=dates, columns=stocks).fillna(True).to_numpy(
            dtype=bool,
            copy=False,
        )
        return buylimit, selllimit

    def calc_result(
        self,
        factor_value: pd.DataFrame,
        profile_id: str = DEFAULT_PROFILE_ID,
        apply_suspend_mask: bool = True,
        apply_limit_mask: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(factor_value, pd.DataFrame):
            raise ValueError("factor_value must be DataFrame")

        profile = PROFILES.get(profile_id)
        if profile is None:
            raise ValueError("profile id not exists")

        buypoint = profile["buypoint"]
        buylag = profile["buylag"]
        sellpoint = profile["sellpoint"]
        selllag = profile["selllag"]
        if buylag > selllag:
            raise ValueError("sell lag cannot be less than buy lag")
        if buylag == selllag and (buypoint == sellpoint or (buypoint == "closes" and sellpoint == "opens")):
            raise ValueError("buy point must before sell point")
        universe_name = profile["universe"]
        qt = profile["qt"]

        if buypoint == "opens" and sellpoint == "opens" and buylag == 1 and selllag == 2:
            mode = 1
        elif buypoint == "closes" and sellpoint == "closes" and buylag == 1 and selllag == 2:
            mode = 2
        else:
            raise ValueError("unsupported profile")

        universe = self._get_data(universe_name).astype(bool)
        factor_value, universe = factor_value.align(universe, join="inner", axis=None)
        factor_value = factor_value.where(universe)
        factor_df = factor_value.shift(1).iloc[1:]

        dates = factor_df.index
        stocks = factor_df.columns
        nd = len(dates)
        ns = len(stocks)
        fv1 = factor_df.to_numpy(dtype=float, copy=False)
        fv2 = factor_df.shift(1).to_numpy(dtype=float, copy=False)
        fr1 = cs_rank_2d(fv1, pct=True, ascending=True)
        fr2 = cs_rank_2d(fv2, pct=True, ascending=True)
        tradables = self._resolve_tradables(dates, stocks, apply_suspend_mask)
        buylimit, selllimit = self._resolve_limit_masks(dates, stocks, mode, apply_limit_mask)

        if mode == 1:
            c2o = self._get_c2o().reindex(index=dates, columns=stocks).to_numpy(dtype=float, copy=False)
            o2c = self._get_o2c().reindex(index=dates, columns=stocks).to_numpy(dtype=float, copy=False)
            o2o = self._get_o2o().reindex(index=dates, columns=stocks).to_numpy(dtype=float, copy=False)
            rv = o2o
        else:
            c2c = self._get_c2c().reindex(index=dates, columns=stocks).to_numpy(dtype=float, copy=False)
            rv = c2c

        result_dict = {}

        target = fr1 >= 1 - qt
        if mode == 1:
            rets_c2o = np.zeros(nd, dtype=float)
            rets_o2c = np.zeros(nd, dtype=float)
            turnovers = np.zeros(nd, dtype=float)
            nums = np.zeros(nd, dtype=float)
            prev = np.zeros(ns, dtype=bool)
            for i in range(nd):
                tgt = target[i]
                buyables = tradables[i] & (~buylimit[i])
                sellables = tradables[i] & (~selllimit[i])
                to_buy = (~prev) & tgt & buyables
                to_sell = prev & (~tgt) & sellables
                real = (prev & (~to_sell)) | to_buy
                n_prev = prev.sum()
                n_real = real.sum()
                if n_prev > 0:
                    rets_c2o[i] = cs_mean_1d(c2o[i, prev])
                    turnovers[i] = (to_buy.sum() + to_sell.sum()) / n_prev
                if n_real > 0:
                    rets_o2c[i] = cs_mean_1d(o2c[i, real])
                    nums[i] = n_real
                prev = real
            rets = (1 + rets_c2o) * (1 + rets_o2c) - 1
        else:
            rets = np.zeros(nd, dtype=float)
            turnovers = np.zeros(nd, dtype=float)
            nums = np.zeros(nd, dtype=float)
            prev = np.zeros(ns, dtype=bool)
            for i in range(nd):
                tgt = target[i]
                buyables = tradables[i] & (~buylimit[i])
                sellables = tradables[i] & (~selllimit[i])
                to_buy = (~prev) & tgt & buyables
                to_sell = prev & (~tgt) & sellables
                real = (prev & (~to_sell)) | to_buy
                n_prev = prev.sum()
                n_real = real.sum()
                if n_prev > 0:
                    rets[i] = cs_mean_1d(c2c[i, prev])
                    turnovers[i] = (to_buy.sum() + to_sell.sum()) / n_prev
                if n_real > 0:
                    nums[i] = n_real
                prev = real
        result_dict["long_rets"] = pd.Series(rets, index=dates, dtype=float)
        result_dict["long_turnovers"] = pd.Series(turnovers, index=dates, dtype=float)
        result_dict["long_nums"] = pd.Series(nums, index=dates, dtype=float)

        target = fr1 <= qt
        if mode == 1:
            rets_c2o = np.zeros(nd, dtype=float)
            rets_o2c = np.zeros(nd, dtype=float)
            turnovers = np.zeros(nd, dtype=float)
            nums = np.zeros(nd, dtype=float)
            prev = np.zeros(ns, dtype=bool)
            for i in range(nd):
                tgt = target[i]
                openables = tradables[i] & (~selllimit[i])
                closeables = tradables[i] & (~buylimit[i])
                to_open = (~prev) & tgt & openables
                to_close = prev & (~tgt) & closeables
                real = (prev & (~to_close)) | to_open
                n_prev = prev.sum()
                n_real = real.sum()
                if n_prev > 0:
                    rets_c2o[i] = -cs_mean_1d(c2o[i, prev])
                    turnovers[i] = (to_open.sum() + to_close.sum()) / n_prev
                if n_real > 0:
                    rets_o2c[i] = -cs_mean_1d(o2c[i, real])
                    nums[i] = n_real
                prev = real
            rets = (1 + rets_c2o) * (1 + rets_o2c) - 1
        else:
            rets = np.zeros(nd, dtype=float)
            turnovers = np.zeros(nd, dtype=float)
            nums = np.zeros(nd, dtype=float)
            prev = np.zeros(ns, dtype=bool)
            for i in range(nd):
                tgt = target[i]
                openables = tradables[i] & (~selllimit[i])
                closeables = tradables[i] & (~buylimit[i])
                to_open = (~prev) & tgt & openables
                to_close = prev & (~tgt) & closeables
                real = (prev & (~to_close)) | to_open
                n_prev = prev.sum()
                n_real = real.sum()
                if n_prev > 0:
                    rets[i] = -cs_mean_1d(c2c[i, prev])
                    turnovers[i] = (to_open.sum() + to_close.sum()) / n_prev
                if n_real > 0:
                    nums[i] = n_real
                prev = real
        result_dict["short_rets"] = pd.Series(rets, index=dates, dtype=float)
        result_dict["short_turnovers"] = pd.Series(turnovers, index=dates, dtype=float)
        result_dict["short_nums"] = pd.Series(nums, index=dates, dtype=float)

        group_rets = {}
        for i in range(10):
            low = i / 10
            high = (i + 1) / 10 if i != 9 else 1 + 1e-9
            group_mask = (fr2 >= low) & (fr2 < high)
            group_rv = np.where(group_mask, rv, np.nan)
            group_rets[f"group_{i}"] = pd.Series(cs_mean_2d(group_rv), index=dates, dtype=float)
        result_dict["group_rets"] = pd.DataFrame(group_rets)

        mask = np.isfinite(fv2) & np.isfinite(rv)
        fv_for_rankic = np.where(mask, fv2, np.nan)
        rv_for_rankic = np.where(mask, rv, np.nan)
        rankics = cs_corr_2d(
            cs_rank_2d(fv_for_rankic, pct=True, ascending=True),
            cs_rank_2d(rv_for_rankic, pct=True, ascending=True),
        )
        result_dict["rankics"] = pd.Series(rankics, index=dates, dtype=float)
        result_dict["coverages"] = factor_value.notna().sum(axis=1) / universe.sum(axis=1)
        return result_dict
