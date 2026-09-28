import os
import sys
from bisect import bisect_left,insort

import numpy as np
import pandas as pd
from tqdm import tqdm
from .safe_pickle_writer import SafePickleWriter


SOURCE_CUTOFF = pd.Timestamp("2010-01-05")
OUTPUT_CUTOFF = pd.Timestamp("2015-01-01")
INCREMENTAL_WARMUP_DAYS = 970
FACTOR_NAMES = [
    "ep",
    "bp",
    "roe",
    "roe_trend",
    "profit_margin",
    "cash_coverage",
    "leverage",
    "asset_growth",
    "revenue_growth",
    "div_yield_ttm",
    "div_yield_yoy",
    "div_yield_stability",
    "payout_ratio",
    "consecutive_div_years",
    "earnings_growth",
    "roe_stability",
    "net_margin",
    "asset_turnover",
    "operating_cost_ratio",
    "cost_ratio_trend",
    "long_term_debt_ratio",
]


class FundamentalUpdater:
    def __init__(self,
                 source_dir:str,
                 fundamental_input_dir:str,
                 base_dir:str):
        self.source_dir = source_dir
        self.fundamental_input_dir = fundamental_input_dir
        self.base_dir = base_dir
        self.dst = os.path.join(base_dir,"fundamental_factors","raw")


    def _load_pkl(self, name: str) -> pd.DataFrame:
        fp = os.path.join(self.source_dir, f"{name}.pkl")
        if not os.path.isfile(fp):
            print(f"ERROR: file not found: {fp}", file=sys.stderr)
            sys.exit(1)
        print(f"Load {name}: {fp}")
        df = pd.read_pickle(fp)
        if hasattr(df, "index") and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index.astype(str))
        if hasattr(df, "sort_index"):
            df = df.sort_index()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.loc[SOURCE_CUTOFF:]
        return df

    def _load_fundamentals(self) -> pd.DataFrame:
        fp = os.path.join(self.fundamental_input_dir,"fundamentals.pkl")
        if not os.path.isfile(fp):
            print(f"ERROR: file not found: {fp}", file=sys.stderr)
            sys.exit(1)

        print(f"Load fundamentals: {fp}")
        fund = pd.read_pickle(fp)
        cols = [
            "code", "date", "quarter", "updateDate",
            "np", "revenue", "op",
            "totalAssets", "totalLiab", "totalOwnerEquity",
            "operatingTotalCost", "longLiab",
        ]
        fund = fund[cols].copy()
        fund["date"] = pd.to_datetime(fund["date"].astype(str))
        fund["updateDate"] = pd.to_datetime(fund["updateDate"].astype(str))

        n_before = len(fund)
        fund = fund.dropna(subset=["code", "date", "updateDate"])
        fund = fund[fund["date"] >= SOURCE_CUTOFF].copy()
        fund = fund[fund["updateDate"] >= SOURCE_CUTOFF].copy()
        # fundamentals.pkl 里存在未来报告期的 0 值占位行，必须先过滤。
        fund = fund[fund["updateDate"] >= fund["date"]].copy()
        print(f"Filter future placeholders: {n_before} -> {len(fund)}")

        fund["quarter"] = fund["quarter"].fillna(fund["date"].dt.quarter).astype(int)
        fund["fiscal_year"] = fund["date"].dt.year.astype(int)

        # 同一股票、同一公告日、同一报告期只保留最后一条。
        fund = (
            fund.sort_values(["code", "updateDate", "date"])
            .drop_duplicates(subset=["code", "updateDate", "date"], keep="last")
            .reset_index(drop=True)
        )
        print(f"Prepared fundamentals: {fund.shape}")
        return fund

    def _quarter_end(self, dt: pd.Timestamp, offset: int = 0) -> pd.Timestamp:
        return (dt.to_period("Q") + offset).to_timestamp(how="end").normalize()

    def _next_trading_day(
        self,
        update_dates: pd.Series,
        trading_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        # 统一映射到“下一个交易日”，避免盘后/周末公告造成前视。
        pos = trading_dates.searchsorted(pd.to_datetime(update_dates), side="right")
        out = np.full(len(update_dates), np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = pos < len(trading_dates)
        out[valid] = trading_dates.values[pos[valid]]
        return pd.Series(pd.to_datetime(out), index=update_dates.index)

    def _safe_ratio(
        self,
        numer: float,
        denom: float,
        lower: float | None = None,
        upper: float | None = None,
    ) -> float:
        if pd.isna(numer) or pd.isna(denom) or denom == 0:
            return np.nan
        value = numer / denom
        if not np.isfinite(value):
            return np.nan
        lo = -np.inf if lower is None else lower
        hi = np.inf if upper is None else upper
        return float(np.clip(value, lo, hi))

    def _safe_growth(
        self,
        current: float,
        previous: float,
        lower: float,
        upper: float,
        abs_denom: bool = False,
    ) -> float:
        if pd.isna(current) or pd.isna(previous):
            return np.nan
        denom = abs(previous) if abs_denom else previous
        if denom == 0:
            return np.nan
        value = (current - previous) / denom
        if not np.isfinite(value):
            return np.nan
        return float(np.clip(value, lower, upper))

    def _refresh_snapshot_cache(
        self,
        report_dates: list[pd.Timestamp],
        state: dict[pd.Timestamp, dict[str, float]],
        single: dict[str, dict[pd.Timestamp, float]],
        ttm: dict[str, dict[pd.Timestamp, float]],
        roe_hist: dict[pd.Timestamp, float],
        start_dt: pd.Timestamp,
    ) -> None:
        flow_fields = ("np", "revenue", "op", "operatingTotalCost")
        start_idx = bisect_left(report_dates, start_dt)

        for report_dt in report_dates[start_idx:]:
            cur = state[report_dt]
            prev_dt = self._quarter_end(report_dt, -1)
            prev = state.get(prev_dt)
            for fld in flow_fields:
                cur_val = cur.get(fld, np.nan)
                if pd.isna(cur_val):
                    single_val = np.nan
                elif int(cur["quarter"]) == 1:
                    single_val = cur_val
                elif (
                    prev is not None
                    and int(prev["fiscal_year"]) == int(cur["fiscal_year"])
                    and pd.notna(prev.get(fld, np.nan))
                ):
                    single_val = cur_val - prev[fld]
                else:
                    single_val = np.nan
                single[fld][report_dt] = single_val

        for report_dt in report_dates[start_idx:]:
            need_dates = [self._quarter_end(report_dt, -i) for i in range(4)]
            for fld in flow_fields:
                vals = [single[fld].get(dt, np.nan) for dt in need_dates]
                ttm[fld][report_dt] = float(np.sum(vals)) if all(pd.notna(v) for v in vals) else np.nan
            equity = state[report_dt].get("totalOwnerEquity", np.nan)
            roe_hist[report_dt] = self._safe_ratio(ttm["np"].get(report_dt, np.nan), equity, -5.0, 5.0)

    def _build_snapshot_from_cache(
        self,
        code: str,
        update_dt: pd.Timestamp,
        effective_dt: pd.Timestamp,
        report_dates: list[pd.Timestamp],
        state: dict[pd.Timestamp, dict[str, float]],
        ttm: dict[str, dict[pd.Timestamp, float]],
        roe_hist: dict[pd.Timestamp, float],
    ) -> dict[str, object]:
        if not report_dates:
            return {}

        latest_dt = report_dates[-1]
        prev_year_dt = self._quarter_end(latest_dt, -4)
        latest = state[latest_dt]
        prev_year = state.get(prev_year_dt, {})

        np_ttm = ttm["np"].get(latest_dt, np.nan)
        revenue_ttm = ttm["revenue"].get(latest_dt, np.nan)
        op_ttm = ttm["op"].get(latest_dt, np.nan)
        np_ttm_prev = ttm["np"].get(prev_year_dt, np.nan)
        revenue_ttm_prev = ttm["revenue"].get(prev_year_dt, np.nan)

        equity = latest.get("totalOwnerEquity", np.nan)
        total_assets = latest.get("totalAssets", np.nan)
        total_liab = latest.get("totalLiab", np.nan)

        roe = self._safe_ratio(np_ttm, equity, -2.0, 2.0)
        roe_prev = roe_hist.get(prev_year_dt, np.nan)
        roe_trend = np.nan if pd.isna(roe) or pd.isna(roe_prev) else float(roe - roe_prev)

        roe_vals = [roe_hist[dt] for dt in report_dates if pd.notna(roe_hist[dt])]
        roe_stability = np.nan
        if len(roe_vals) >= 8:
            roe_stability = -float(np.std(roe_vals[-12:], ddof=1))

        oc_ttm = ttm["operatingTotalCost"].get(latest_dt, np.nan)
        oc_ttm_prev = ttm["operatingTotalCost"].get(prev_year_dt, np.nan)
        long_liab = latest.get("longLiab", np.nan)

        net_margin = self._safe_ratio(np_ttm, revenue_ttm, -5.0, 5.0)
        asset_turnover = self._safe_ratio(revenue_ttm, total_assets, 0.0, 20.0)
        operating_cost_ratio = self._safe_ratio(oc_ttm, revenue_ttm, 0.0, 5.0)
        operating_cost_ratio_prev = self._safe_ratio(oc_ttm_prev, revenue_ttm_prev, 0.0, 5.0)
        cost_ratio_trend = np.nan
        if pd.notna(operating_cost_ratio) and pd.notna(operating_cost_ratio_prev):
            cost_ratio_trend = float(operating_cost_ratio - operating_cost_ratio_prev)

        long_term_debt_ratio = self._safe_ratio(long_liab, total_liab, 0.0, 1.0)

        return {
            "code": code,
            "updateDate": update_dt,
            "effectiveDate": effective_dt,
            "reportDate": latest_dt,
            "np_ttm": np_ttm,
            "revenue_ttm": revenue_ttm,
            "op_ttm": op_ttm,
            "totalAssets": total_assets,
            "totalLiab": total_liab,
            "totalOwnerEquity": equity,
            "roe": roe,
            "roe_trend": roe_trend,
            "profit_margin": self._safe_ratio(op_ttm, revenue_ttm, -5.0, 5.0),
            "cash_coverage": self._safe_ratio(op_ttm, total_liab, -5.0, 5.0),
            "leverage": self._safe_ratio(total_liab, total_assets, 0.0, 2.0),
            "asset_growth": self._safe_growth(total_assets, prev_year.get("totalAssets", np.nan), -2.0, 10.0),
            "revenue_growth": self._safe_growth(revenue_ttm, revenue_ttm_prev, -2.0, 10.0),
            "earnings_growth": self._safe_growth(np_ttm, np_ttm_prev, -3.0, 10.0, abs_denom=True),
            "roe_stability": roe_stability,
            "net_margin": net_margin,
            "asset_turnover": asset_turnover,
            "operating_cost_ratio": operating_cost_ratio,
            "cost_ratio_trend": cost_ratio_trend,
            "long_term_debt_ratio": long_term_debt_ratio,
        }

    def _build_event_table(
        self,
        fund: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        fund = fund.copy()
        if "effectiveDate" not in fund.columns:
            fund["effectiveDate"] = self._next_trading_day(fund["updateDate"], trading_dates)
        else:
            fund["effectiveDate"] = pd.to_datetime(fund["effectiveDate"])
        fund = fund[fund["effectiveDate"].notna()].copy()

        rows = []
        code_groups = fund.groupby("code", sort=False)
        for code, code_df in tqdm(code_groups,total=fund["code"].nunique(),desc="build_fundamental_events"):
            state = {}
            report_dates = []
            single = {fld: {} for fld in ["np", "revenue", "op", "operatingTotalCost"]}
            ttm = {fld: {} for fld in ["np", "revenue", "op", "operatingTotalCost"]}
            roe_hist = {}
            code_df = code_df.sort_values(["updateDate", "date"])
            for update_dt, grp in code_df.groupby("updateDate", sort=True):
                effective_dt = grp["effectiveDate"].iloc[0]
                changed_dates = []
                for row in grp.itertuples(index=False):
                    report_dt = pd.Timestamp(row.date)
                    if report_dt not in state:
                        insort(report_dates, report_dt)
                    state[report_dt] = {
                        "quarter": int(row.quarter),
                        "fiscal_year": int(row.fiscal_year),
                        "np": row.np,
                        "revenue": row.revenue,
                        "op": row.op,
                        "totalAssets": row.totalAssets,
                        "totalLiab": row.totalLiab,
                        "totalOwnerEquity": row.totalOwnerEquity,
                        "operatingTotalCost": row.operatingTotalCost,
                        "longLiab": row.longLiab,
                    }
                    changed_dates.append(report_dt)
                self._refresh_snapshot_cache(report_dates,state,single,ttm,roe_hist,min(changed_dates))
                snap = self._build_snapshot_from_cache(code,update_dt,effective_dt,report_dates,state,ttm,roe_hist)
                if snap:
                    rows.append(snap)

        event_df = pd.DataFrame(rows)
        event_df = event_df.sort_values(["effectiveDate", "updateDate", "reportDate", "code"]).reset_index(drop=True)
        print(f"Event snapshots: {event_df.shape}")
        return event_df

    def _panel_from_events(
        self,
        event_df: pd.DataFrame,
        column: str,
        trading_dates: pd.DatetimeIndex,
        all_stocks: pd.Index,
        event_presence: pd.DataFrame = None,
    ) -> pd.DataFrame:
        if event_df.empty:
            return pd.DataFrame(index=trading_dates, columns=all_stocks, dtype=np.float32)

        sentinel = np.float64(9.99e37)
        panel = event_df.pivot_table(
            index="effectiveDate",
            columns="code",
            values=column,
            aggfunc="last",
        )
        panel = panel.reindex(columns=all_stocks)

        # 仅当「该日该股票有财报事件但因子值为 NaN」时阻断前向填充，避免把缺失误填成有效值。
        if event_presence is not None:
            ep = event_presence.reindex(index=panel.index, columns=all_stocks).fillna(False)
            panel = panel.where(~(ep & panel.isna()), sentinel)
        else:
            panel = panel.where(panel.notna(), sentinel)
        all_dates = panel.index.union(trading_dates).sort_values().unique()
        panel = panel.reindex(all_dates).ffill().reindex(trading_dates)
        return panel.replace(sentinel, np.nan).astype(np.float32)

    def _build_panels_from_events(
        self,
        event_df: pd.DataFrame,
        panel_fields: list[tuple[str, str]],
        trading_dates: pd.DatetimeIndex,
        all_stocks: pd.Index,
    ) -> dict[str, pd.DataFrame]:
        if event_df.empty:
            return {name: pd.DataFrame(index=trading_dates, columns=all_stocks, dtype=np.float32) for name, _ in panel_fields}

        sentinel = np.float64(9.99e37)
        columns = [column for _, column in panel_fields]
        event_presence = (
            event_df[["effectiveDate", "code"]]
            .drop_duplicates()
            .assign(_present=True)
            .set_index(["effectiveDate", "code"])["_present"]
            .unstack(fill_value=False)
            .astype(bool)
        )
        event_last = event_df.drop_duplicates(subset=["effectiveDate", "code"], keep="last")
        wide = (
            event_last.set_index(["effectiveDate", "code"])[columns]
            .unstack("code")
            .sort_index()
        )
        all_dates = wide.index.union(trading_dates).sort_values().unique()
        wide = wide.reindex(all_dates)
        event_presence = event_presence.reindex(index=all_dates, columns=all_stocks, fill_value=False).astype(bool)

        panels = {}
        for name, column in tqdm(panel_fields,total=len(panel_fields),desc="build_fundamental_panels"):
            panel = wide[column].reindex(columns=all_stocks)
            panel = panel.where(~(event_presence & panel.isna()), sentinel)
            panels[name] = panel.ffill().reindex(trading_dates).replace(sentinel, np.nan).astype(np.float32)
        return panels

    def _build_ff_factors(
        self,
        trading_dates: pd.DatetimeIndex,
        tradables: pd.DataFrame,
        returns: pd.DataFrame,
        size: pd.DataFrame,
        bp: pd.DataFrame,
        profit_margin: pd.DataFrame,
        asset_growth: pd.DataFrame,
        index_data: pd.DataFrame,
    ) -> pd.DataFrame:
        ff_mkt = index_data["zzhls_h_returns"].reindex(trading_dates).fillna(0.0).astype(np.float32)

        dates_2010 = trading_dates[trading_dates >= pd.Timestamp("2010-01-01")]
        ff_smb = pd.Series(0.0, index=dates_2010, dtype=np.float32)
        ff_hml = pd.Series(0.0, index=dates_2010, dtype=np.float32)
        ff_rmw = pd.Series(0.0, index=dates_2010, dtype=np.float32)
        ff_cma = pd.Series(0.0, index=dates_2010, dtype=np.float32)

        prev_month = None
        port = {}
        for date in tqdm(dates_2010,desc="build_ff_factors"):
            cur_month = date.to_period("M")
            loc = trading_dates.get_loc(date)
            signal_date = trading_dates[loc - 1] if isinstance(loc, int) and loc > 0 else None

            if cur_month != prev_month:
                prev_month = cur_month
                port = {}
                if signal_date is None:
                    continue

                valid = tradables.loc[signal_date]
                valid = valid[valid.astype(bool)].index

                sz = size.loc[signal_date].reindex(valid).dropna()
                if len(sz) >= 40:
                    med = sz.median()
                    port["small"] = sz[sz <= med].index
                    port["big"] = sz[sz > med].index

                bp_row = bp.loc[signal_date].reindex(valid).dropna()
                if len(bp_row) >= 40:
                    q30, q70 = bp_row.quantile([0.3, 0.7])
                    port["high_bp"] = bp_row[bp_row >= q70].index
                    port["low_bp"] = bp_row[bp_row <= q30].index

                pm_row = profit_margin.loc[signal_date].reindex(valid).dropna()
                if len(pm_row) >= 40:
                    q30, q70 = pm_row.quantile([0.3, 0.7])
                    port["robust"] = pm_row[pm_row >= q70].index
                    port["weak"] = pm_row[pm_row <= q30].index

                ag_row = asset_growth.loc[signal_date].reindex(valid).dropna()
                if len(ag_row) >= 40:
                    q30, q70 = ag_row.quantile([0.3, 0.7])
                    port["conservative"] = ag_row[ag_row <= q30].index
                    port["aggressive"] = ag_row[ag_row >= q70].index

            ret = returns.loc[date]

            if "small" in port and "big" in port:
                ff_smb.loc[date] = np.float32(ret.reindex(port["small"]).mean() - ret.reindex(port["big"]).mean())
            if "high_bp" in port and "low_bp" in port:
                ff_hml.loc[date] = np.float32(ret.reindex(port["high_bp"]).mean() - ret.reindex(port["low_bp"]).mean())
            if "robust" in port and "weak" in port:
                ff_rmw.loc[date] = np.float32(ret.reindex(port["robust"]).mean() - ret.reindex(port["weak"]).mean())
            if "conservative" in port and "aggressive" in port:
                ff_cma.loc[date] = np.float32(ret.reindex(port["conservative"]).mean() - ret.reindex(port["aggressive"]).mean())

        return pd.DataFrame(
            {
                "ff_mkt": ff_mkt,
                "ff_smb": ff_smb.reindex(trading_dates, fill_value=0.0),
                "ff_hml": ff_hml.reindex(trading_dates, fill_value=0.0),
                "ff_rmw": ff_rmw.reindex(trading_dates, fill_value=0.0),
                "ff_cma": ff_cma.reindex(trading_dates, fill_value=0.0),
            }
        ).astype(np.float32)

    def build_fundamental_factors(
        self,
        fund: pd.DataFrame,
        tradables: pd.DataFrame,
        mcap: pd.DataFrame,
        closes: pd.DataFrame,
        cash_div: pd.DataFrame,
        returns: pd.DataFrame,
        size: pd.DataFrame,
        index_data: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        trading_dates = pd.DatetimeIndex(tradables.index).sort_values()
        all_stocks = tradables.columns

        event_df = self._build_event_table(fund, trading_dates)

        panel_fields = [
            ("np_ttm","np_ttm"),
            ("revenue_ttm","revenue_ttm"),
            ("op_ttm","op_ttm"),
            ("equity","totalOwnerEquity"),
            ("total_assets","totalAssets"),
            ("total_liab","totalLiab"),
            ("roe","roe"),
            ("roe_trend","roe_trend"),
            ("profit_margin","profit_margin"),
            ("cash_coverage","cash_coverage"),
            ("leverage","leverage"),
            ("asset_growth","asset_growth"),
            ("revenue_growth","revenue_growth"),
            ("earnings_growth","earnings_growth"),
            ("roe_stability","roe_stability"),
            ("net_margin","net_margin"),
            ("asset_turnover","asset_turnover"),
            ("operating_cost_ratio","operating_cost_ratio"),
            ("cost_ratio_trend","cost_ratio_trend"),
            ("long_term_debt_ratio","long_term_debt_ratio"),
        ]
        panels = self._build_panels_from_events(event_df,panel_fields,trading_dates,all_stocks)

        np_ttm = panels["np_ttm"]
        revenue_ttm = panels["revenue_ttm"]
        op_ttm = panels["op_ttm"]
        equity = panels["equity"]
        total_assets = panels["total_assets"]
        total_liab = panels["total_liab"]
        roe = panels["roe"]
        roe_trend = panels["roe_trend"]
        profit_margin = panels["profit_margin"]
        cash_coverage = panels["cash_coverage"]
        leverage = panels["leverage"]
        asset_growth = panels["asset_growth"]
        revenue_growth = panels["revenue_growth"]
        earnings_growth = panels["earnings_growth"]
        roe_stability = panels["roe_stability"]
        net_margin = panels["net_margin"]
        asset_turnover = panels["asset_turnover"]
        operating_cost_ratio = panels["operating_cost_ratio"]
        cost_ratio_trend = panels["cost_ratio_trend"]
        long_term_debt_ratio = panels["long_term_debt_ratio"]

        tradables = tradables.reindex(index=trading_dates, columns=all_stocks).fillna(0).astype(bool)
        mcap = mcap.reindex(index=trading_dates, columns=all_stocks)
        closes = closes.reindex(index=trading_dates, columns=all_stocks)
        cash_div = cash_div.reindex(index=trading_dates, columns=all_stocks).fillna(0.0)
        returns = returns.reindex(index=trading_dates, columns=all_stocks)
        size = size.reindex(index=trading_dates, columns=all_stocks)
        mcap_safe = mcap.replace(0, np.nan)
        close_safe = closes.replace(0, np.nan)

        factors = {}

        # 1. 盈利收益率 EP = TTM净利润 / 总市值
        factors["ep"] = (np_ttm / mcap_safe).replace([np.inf, -np.inf], np.nan).astype(np.float32)

        # 2. 账面市值比 BP = 股东权益 / 总市值
        factors["bp"] = (equity / mcap_safe).replace([np.inf, -np.inf], np.nan).astype(np.float32)

        # 3. 净资产收益率 ROE = TTM净利润 / 股东权益
        factors["roe"] = roe.astype(np.float32)

        # 4. ROE 趋势 = 当前 ROE − 去年同期 ROE
        factors["roe_trend"] = roe_trend.astype(np.float32)

        # 5. 营业利润率 = TTM营业利润 / TTM营业收入
        factors["profit_margin"] = profit_margin.astype(np.float32)

        # 6. 利润偿债覆盖度 = TTM营业利润 / 总负债
        factors["cash_coverage"] = cash_coverage.astype(np.float32)

        # 7. 杠杆率 = 总负债 / 总资产
        factors["leverage"] = leverage.astype(np.float32)

        # 8. 资产增速 = 总资产同比增速
        factors["asset_growth"] = asset_growth.astype(np.float32)

        # 9. 营收增速 = TTM营业收入同比增速
        factors["revenue_growth"] = revenue_growth.astype(np.float32)

        # 10. TTM 股息率 = 过去243个交易日每股现金分红 / 收盘价
        div_ttm_243 = cash_div.rolling(window=243,min_periods=21).sum()
        div_cash_div = cash_div.loc[OUTPUT_CUTOFF:]
        div_close_safe = close_safe.loc[OUTPUT_CUTOFF:]
        div_ttm_243_output = div_cash_div.rolling(window=243,min_periods=21).sum()
        div_yield_ttm = (div_ttm_243_output / div_close_safe).clip(0,0.50)
        factors["div_yield_ttm"] = div_yield_ttm.astype(np.float32)

        # 11. 股息率同比 = 当前 TTM 股息率 − 去年同期 TTM 股息率
        factors["div_yield_yoy"] = (div_yield_ttm - div_yield_ttm.shift(243)).astype(np.float32)

        # 12. 股息率稳定性 = −变异系数（约3年窗口），数值越大表示越稳定
        div_mean = div_yield_ttm.rolling(729, min_periods=243).mean()
        div_std = div_yield_ttm.rolling(729, min_periods=243).std()
        factors["div_yield_stability"] = (-(div_std / (div_mean + 1e-8))).astype(np.float32)

        # 13. 派息率 = 过去243日总分红 / TTM净利润
        total_shares = mcap_safe / close_safe
        annual_total_div = div_ttm_243 * total_shares
        payout_ratio = (annual_total_div / np_ttm.replace(0, np.nan)).clip(0, 2.0)
        factors["payout_ratio"] = payout_ratio.replace([np.inf, -np.inf], np.nan).astype(np.float32)

        # 14. 连续分红年数 = 向前连续有分红的年数
        max_years = 15
        annual_flags = []
        for yr in range(max_years):
            start = yr * 243
            end = (yr + 1) * 243
            if end > len(cash_div):
                break
            window_sum = cash_div.shift(start).rolling(window=243, min_periods=21).sum()
            annual_flags.append((window_sum > 0).astype(np.float32))

        consecutive_div_years = pd.DataFrame(0.0, index=trading_dates, columns=all_stocks, dtype=np.float32)
        if annual_flags:
            consecutive_div_years = annual_flags[0].astype(np.float32)
            for yr in range(1, len(annual_flags)):
                consecutive_div_years = consecutive_div_years + annual_flags[yr] * (consecutive_div_years >= yr).astype(np.float32)
        factors["consecutive_div_years"] = consecutive_div_years.astype(np.float32)

        # 15. 盈利增速 = TTM净利润同比增速
        factors["earnings_growth"] = earnings_growth.astype(np.float32)

        # 16. ROE 稳定性 = −std(最近12个季度 ROE_TTM)，数值越大表示越稳定
        factors["roe_stability"] = roe_stability.astype(np.float32)

        # 17. 净利率 = TTM净利润 / TTM营业收入
        factors["net_margin"] = net_margin.astype(np.float32)

        # 18. 资产周转率 = TTM营业收入 / 总资产
        factors["asset_turnover"] = asset_turnover.astype(np.float32)

        # 19. 营业成本率 = TTM营业总成本 / TTM营业收入（越低越好）
        factors["operating_cost_ratio"] = operating_cost_ratio.astype(np.float32)

        # 20. 成本率趋势 = 当前成本率 − 去年同期成本率
        factors["cost_ratio_trend"] = cost_ratio_trend.astype(np.float32)

        # 21. 长期负债占比 = 长期负债 / 总负债（反映债务结构）
        factors["long_term_debt_ratio"] = long_term_debt_ratio.astype(np.float32)

        ff_df = self._build_ff_factors(
            trading_dates=trading_dates,
            tradables=tradables,
            returns=returns,
            size=size,
            bp=factors["bp"],
            profit_margin=factors["profit_margin"],
            asset_growth=factors["asset_growth"],
            index_data=index_data.sort_index(),
        )
        return factors, ff_df

    def _get_output_path(self,name:str)->str:
        return os.path.join(self.dst,f"{name}.pkl")

    def _load_existing_output(self,name:str)->pd.DataFrame|None:
        fp = self._get_output_path(name)
        return pd.read_pickle(fp) if os.path.isfile(fp) else None

    def _can_incremental(self)->bool:
        return all(os.path.isfile(self._get_output_path(name)) for name in FACTOR_NAMES+["ff_factors"])

    def _get_saved_end_date(self)->pd.Timestamp|None:
        ff_df = self._load_existing_output("ff_factors")
        if ff_df is None or len(ff_df.index)==0:
            return None
        return pd.Timestamp(pd.to_datetime(ff_df.index).max())

    def _prepare_incremental_fund(
        self,
        fund:pd.DataFrame,
        trading_dates:pd.DatetimeIndex,
        calc_start:pd.Timestamp,
    ) -> pd.DataFrame:
        fund = fund.copy()
        fund["effectiveDate"] = self._next_trading_day(fund["updateDate"], trading_dates)
        fund = fund[fund["effectiveDate"].notna()].copy()
        history = (
            fund.loc[fund["effectiveDate"] < calc_start]
            .sort_values(["code", "date", "updateDate"])
            .drop_duplicates(subset=["code", "date"], keep="last")
        )
        tail = fund.loc[fund["effectiveDate"] >= calc_start]
        out = pd.concat([history, tail], ignore_index=True)
        return out.sort_values(["code", "updateDate", "date"]).reset_index(drop=True)

    def _merge_saved_output(
        self,
        old:pd.DataFrame|None,
        new:pd.DataFrame,
        merge_start:pd.Timestamp,
    ) -> pd.DataFrame:
        if old is None or len(old)==0:
            return new.sort_index()
        merged = pd.concat([
            old.loc[old.index < merge_start],
            new.loc[new.index >= merge_start],
        ]).sort_index()
        return merged.loc[~merged.index.duplicated(keep="last")]

    def _calc_consecutive_div_years(self,cash_div:pd.DataFrame)->pd.DataFrame:
        trading_dates = cash_div.index
        all_stocks = cash_div.columns
        annual_flags = []
        for yr in range(15):
            start = yr*243
            end = (yr+1)*243
            if end > len(cash_div):
                break
            window_sum = cash_div.shift(start).rolling(window=243,min_periods=21).sum()
            annual_flags.append((window_sum > 0).astype(np.float32))

        consecutive_div_years = pd.DataFrame(0.0,index=trading_dates,columns=all_stocks,dtype=np.float32)
        if annual_flags:
            consecutive_div_years = annual_flags[0].astype(np.float32)
            for yr in range(1,len(annual_flags)):
                consecutive_div_years = consecutive_div_years+annual_flags[yr]*(consecutive_div_years >= yr).astype(np.float32)
        return consecutive_div_years.astype(np.float32)

    def _build_ff_roll(self,ff_df:pd.DataFrame)->pd.DataFrame:
        ff_roll = ff_df.copy()
        for col in ["ff_mkt", "ff_smb", "ff_hml", "ff_rmw", "ff_cma"]:
            ff_roll[f"{col}_20d"] = ff_roll[col].rolling(20).sum().astype(np.float32)
            ff_roll[f"{col}_60d"] = ff_roll[col].rolling(60).sum().astype(np.float32)
        return ff_roll.astype(np.float32)

    def update(self):

        output_dir = self.dst
        os.makedirs(output_dir, exist_ok=True)

        fund = self._load_fundamentals()
        tradables = self._load_pkl("tradables")
        mcap = self._load_pkl("total_market_caps")
        closes = self._load_pkl("closes")
        cash_div = self._load_pkl("cash_dividends")
        returns = self._load_pkl("ctc_returns")
        size = self._load_pkl("size")
        index_data = self._load_pkl("index_data")

        trading_dates = pd.DatetimeIndex(tradables.index).sort_values()
        latest_trading_date = trading_dates[-1]

        use_incremental = self._can_incremental()
        saved_end_date = self._get_saved_end_date() if use_incremental else None
        if saved_end_date is not None and saved_end_date >= latest_trading_date:
            print(f"Already up to date: {saved_end_date:%Y-%m-%d}")
            return

        if use_incremental and saved_end_date is not None:
            merge_pos = trading_dates.searchsorted(saved_end_date)
            if merge_pos >= len(trading_dates):
                print(f"Already up to date: {saved_end_date:%Y-%m-%d}")
                return
            merge_start = trading_dates[merge_pos]
            calc_start = trading_dates[max(0,merge_pos-INCREMENTAL_WARMUP_DAYS)]
            tail_dates = trading_dates[trading_dates >= calc_start]
            all_stocks = tradables.columns

            print(f"Incremental update: merge_start={merge_start:%Y-%m-%d}, calc_start={calc_start:%Y-%m-%d}")
            fund_tail = self._prepare_incremental_fund(fund,trading_dates,calc_start)
            print("Build incremental fundamental factors ...")
            factors_tail, ff_tail = self.build_fundamental_factors(
                fund=fund_tail,
                tradables=tradables.reindex(index=tail_dates,columns=all_stocks),
                mcap=mcap.reindex(index=tail_dates,columns=all_stocks),
                closes=closes.reindex(index=tail_dates,columns=all_stocks),
                cash_div=cash_div.reindex(index=tail_dates,columns=all_stocks),
                returns=returns.reindex(index=tail_dates,columns=all_stocks),
                size=size.reindex(index=tail_dates,columns=all_stocks),
                index_data=index_data.reindex(index=tail_dates),
            )
            consecutive_div_years = self._calc_consecutive_div_years(
                cash_div.reindex(index=trading_dates,columns=all_stocks).fillna(0.0)
            )

            factors = {}
            for name in FACTOR_NAMES:
                old_df = self._load_existing_output(name)
                new_df = consecutive_div_years if name=="consecutive_div_years" else factors_tail[name]
                factors[name] = self._merge_saved_output(old_df,new_df,merge_start).astype(np.float32)

            ff_df = self._merge_saved_output(
                self._load_existing_output("ff_factors"),
                ff_tail,
                merge_start,
            ).astype(np.float32)
        else:
            print("Build fundamental factors ...")
            factors, ff_df = self.build_fundamental_factors(
                fund=fund,
                tradables=tradables,
                mcap=mcap,
                closes=closes,
                cash_div=cash_div,
                returns=returns,
                size=size,
                index_data=index_data,
            )

        for name, df in factors.items():
            df = df.loc[OUTPUT_CUTOFF:]
            out_fp = os.path.join(output_dir, f"{name}.pkl")
            SafePickleWriter.safe_to_pickle(df.astype(np.float32),out_fp)
            print(f"Save {name}: {out_fp}")

        ff_path = os.path.join(output_dir, "ff_factors.pkl")
        SafePickleWriter.safe_to_pickle(ff_df.loc[OUTPUT_CUTOFF:].astype(np.float32),ff_path)
        print(f"Save ff_factors: {ff_path}")

        ff_roll = self._build_ff_roll(ff_df).loc[OUTPUT_CUTOFF:]
        ff_roll_path = os.path.join(output_dir, "ff_factors_rolling.pkl")
        SafePickleWriter.safe_to_pickle(ff_roll.astype(np.float32),ff_roll_path)
        print(f"Save ff_factors_rolling: {ff_roll_path}")
        print("Done.")



