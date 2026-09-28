import pandas as pd
import numpy as np


class FactorWeeklyReporter:
    def __init__(self,interactor,portrait_store,dm):
        self.interactor = interactor
        self.portrait_store = portrait_store
        self.dm = dm
        self._index_weekly_rets:pd.Series|None = None

    # ----------------------------------------------------------
    # index benchmark
    # ----------------------------------------------------------
    def _ensure_index_weekly_rets(self):
        if self._index_weekly_rets is not None:
            return
        index_data = self.dm.get_data("index_data")
        daily_rets = index_data["zz1000s_returns"]
        daily_rets.index = pd.to_datetime(daily_rets.index)
        week_labels = daily_rets.index.to_period("W")
        weekly = {}
        for period,group in daily_rets.groupby(week_labels):
            week_end = group.index[-1]
            weekly[week_end] = (1+group).prod()-1
        self._index_weekly_rets = pd.Series(weekly).sort_index()

    def _get_index_week_ret(self,week_end)->float:
        if self._index_weekly_rets is None:
            return np.nan
        if week_end in self._index_weekly_rets.index:
            return self._index_weekly_rets[week_end]
        return np.nan

    # ----------------------------------------------------------
    # generate
    # ----------------------------------------------------------
    def generate(self,end_date:str|None=None)->dict:
        factors = self.interactor.list_factors()
        factor_names = self.interactor.filter_names(tier="selected")

        self._ensure_index_weekly_rets()

        valid_names = []
        rows = []
        alerts = []
        week_end_ts = None

        for name in factor_names:
            if not self.portrait_store.has(name):
                continue
            portrait = self.portrait_store.load(name)
            raw = portrait.get("raw",{})
            weekly_perf = raw.get("weekly") if isinstance(raw,dict) else None
            if weekly_perf is None or weekly_perf.empty:
                continue

            # locate current week
            if end_date is not None:
                target = pd.Timestamp(end_date)
                if target not in weekly_perf.index:
                    continue
                cur_idx = weekly_perf.index.get_loc(target)
            else:
                cur_idx = len(weekly_perf)-1

            cur = weekly_perf.iloc[cur_idx]
            hist = weekly_perf.iloc[:cur_idx]
            prev = weekly_perf.iloc[cur_idx-1] if cur_idx>0 else None

            if week_end_ts is None:
                week_end_ts = weekly_perf.index[cur_idx]

            # excess return over index
            idx_ret = self._get_index_week_ret(weekly_perf.index[cur_idx])
            net_excess = cur["long_netret"]-idx_ret if not np.isnan(idx_ret) and "long_netret" in cur.index else np.nan

            prev_idx_ret = self._get_index_week_ret(weekly_perf.index[cur_idx-1]) if cur_idx>0 else np.nan
            prev_net_excess = (prev["long_netret"]-prev_idx_ret) if prev is not None and not np.isnan(prev_idx_ret) and "long_netret" in prev.index else np.nan

            # historical excess for percentile
            hist_net_excess = pd.Series(dtype=float)
            if not hist.empty and "long_netret" in hist.columns:
                hist_idx_rets = pd.Series(
                    [self._get_index_week_ret(d) for d in hist.index],
                    index=hist.index,
                )
                hist_net_excess = hist["long_netret"]-hist_idx_rets
                hist_net_excess = hist_net_excess.dropna()

            net_excess_pct = (hist_net_excess<net_excess).mean()*100 if len(hist_net_excess)>0 and not np.isnan(net_excess) else np.nan

            # percentiles for ls_netret and rankic
            pcts = self._calc_percentiles(cur,hist)

            row = {
                "idx_ret":idx_ret,
                "net_excess":net_excess,
                "net_excess_pct":net_excess_pct,
                "ls_netret":cur.get("ls_netret",np.nan),
                "ls_netret_pct":pcts.get("ls_netret_pct",np.nan),
                "rankic":cur.get("rankic",np.nan),
                "rankic_pct":pcts.get("rankic_pct",np.nan),
                "long_turnover":cur.get("long_turnover",np.nan),
                "coverage":cur.get("coverage",np.nan),
                "prev_net_excess":prev_net_excess,
                "prev_ls_netret":prev["ls_netret"] if prev is not None and "ls_netret" in prev.index else np.nan,
                "prev_rankic":prev["rankic"] if prev is not None else np.nan,
                "tag":factors[name].get("tag",""),
                "category":factors[name].get("category",""),
            }
            valid_names.append(name)
            rows.append(row)

            factor_alerts = self._detect_alerts(name,row,hist_net_excess)
            alerts.extend(factor_alerts)

        if not rows:
            return {"meta":{},"summary":{},"details":pd.DataFrame(),"alerts":[],"tag_summary":pd.DataFrame(),"category_summary":pd.DataFrame()}

        details = pd.DataFrame(rows,index=valid_names)
        details = details.sort_values("net_excess",ascending=False,na_position="last")
        details.index.name = "factor_name"

        # summary
        summary = {
            "idx_ret":details["idx_ret"].iloc[0] if "idx_ret" in details.columns else np.nan,
            "avg_net_excess":details["net_excess"].mean(),
            "avg_net_excess_pct":details["net_excess_pct"].mean(),
            "avg_ls_netret":details["ls_netret"].mean(),
            "avg_ls_netret_pct":details["ls_netret_pct"].mean(),
            "avg_rankic":details["rankic"].mean(),
            "avg_rankic_pct":details["rankic_pct"].mean(),
            "above_75_count":int((details["net_excess_pct"]>=75).sum()),
            "below_25_count":int((details["net_excess_pct"]<=25).sum()),
        }

        # group summaries
        tag_summary = self._build_group_summary(details,"tag")
        category_summary = self._build_group_summary(details,"category")

        meta = {
            "week_end":week_end_ts,
            "factor_count":len(details),
        }

        return {
            "meta":meta,
            "summary":summary,
            "details":details,
            "alerts":alerts,
            "tag_summary":tag_summary,
            "category_summary":category_summary,
        }

    # ----------------------------------------------------------
    # internals
    # ----------------------------------------------------------
    def _calc_percentiles(self,cur:pd.Series,hist:pd.DataFrame)->dict:
        keys = ["ls_netret","rankic"]
        pcts = {}
        for k in keys:
            if k not in hist.columns or hist[k].dropna().empty:
                pcts[f"{k}_pct"] = np.nan
            else:
                pcts[f"{k}_pct"] = (hist[k].dropna()<cur[k]).mean()*100
        return pcts

    def _build_group_summary(self,details:pd.DataFrame,group_col:str)->pd.DataFrame:
        if group_col not in details.columns or not details[group_col].str.len().any():
            return pd.DataFrame()
        grouped = details.groupby(group_col)
        df = pd.DataFrame({
            "count":grouped.size(),
            "avg_net_excess":grouped["net_excess"].mean(),
            "avg_ls_netret":grouped["ls_netret"].mean(),
            "avg_rankic":grouped["rankic"].mean(),
            "avg_net_excess_pct":grouped["net_excess_pct"].mean(),
        })
        return df.sort_values("avg_net_excess",ascending=False)

    def _detect_alerts(self,name:str,row:dict,hist_net_excess:pd.Series)->list[dict]:
        alerts = []

        # 1. extreme poor excess
        net_excess_pct = row.get("net_excess_pct",np.nan)
        if not np.isnan(net_excess_pct) and net_excess_pct<10:
            alerts.append({
                "factor_name":name,
                "type":"超额极差",
                "message":f"本周净超额{row['net_excess']:+.2%}, 历史分位{net_excess_pct:.0f}%",
            })

        # 2. consecutive IC failure
        cur_ic = row.get("rankic",np.nan)
        prev_ic = row.get("prev_rankic",np.nan)
        if not np.isnan(cur_ic) and not np.isnan(prev_ic) and cur_ic<0 and prev_ic<0:
            alerts.append({
                "factor_name":name,
                "type":"IC连续失效",
                "message":f"本周IC={cur_ic:.3f}, 上周IC={prev_ic:.3f}",
            })

        # 3. consecutive excess failure
        cur_excess = row.get("net_excess",np.nan)
        prev_excess = row.get("prev_net_excess",np.nan)
        if not np.isnan(cur_excess) and not np.isnan(prev_excess) and cur_excess<0 and prev_excess<0:
            alerts.append({
                "factor_name":name,
                "type":"超额连续失效",
                "message":f"本周净超额{cur_excess:+.2%}, 上周净超额{prev_excess:+.2%}",
            })

        # 4. coverage too low
        coverage = row.get("coverage",np.nan)
        if not np.isnan(coverage) and coverage<0.5:
            alerts.append({
                "factor_name":name,
                "type":"覆盖过低",
                "message":f"本周覆盖率{coverage:.0%}",
            })

        return alerts

    # ----------------------------------------------------------
    # render
    # ----------------------------------------------------------
    def render_txt(self,report_data:dict)->str:
        meta = report_data["meta"]
        summary = report_data["summary"]
        details = report_data["details"]
        alerts = report_data["alerts"]
        tag_summary = report_data.get("tag_summary",pd.DataFrame())
        cat_summary = report_data.get("category_summary",pd.DataFrame())

        if details.empty:
            return "无可用数据生成周报"

        lines = []
        sep = "="*60

        # header
        lines.append(sep)
        lines.append(f"因子周报  截至 {meta['week_end'].strftime('%Y-%m-%d') if meta.get('week_end') else 'N/A'}")
        lines.append(sep)
        idx_ret = summary.get("idx_ret",np.nan)
        idx_str = f"{idx_ret:+.2%}" if not np.isnan(idx_ret) else "N/A"
        lines.append(f"因子总数: {meta.get('factor_count',0)}")
        lines.append(f"中证1000本周: {idx_str}")
        lines.append(f"平均净超额: {summary['avg_net_excess']:+.2%}  (历史分位 {summary['avg_net_excess_pct']:.0f}%)")
        lines.append(f"平均多空净值: {summary['avg_ls_netret']:+.2%}  (历史分位 {summary['avg_ls_netret_pct']:.0f}%)")
        lines.append(f"平均RankIC: {summary['avg_rankic']:.4f}  (历史分位 {summary['avg_rankic_pct']:.0f}%)")
        lines.append(f"超额优于75分位: {summary['above_75_count']}个 | 低于25分位: {summary['below_25_count']}个")
        lines.append("")

        # top / bottom
        rank_cols = ["net_excess","net_excess_pct","ls_netret","ls_netret_pct","rankic","rankic_pct"]
        rank_headers = ["net_excess","excess_pct","ls_netret","ls_pct","rankic","ic_pct"]
        n_show = min(5,len(details))

        lines.append(">>> 本周最佳 TOP5")
        lines.append(self._fmt_table(details.head(n_show),rank_cols,rank_headers))
        lines.append("")

        lines.append(">>> 本周最差 BOTTOM5")
        lines.append(self._fmt_table(details.tail(n_show).iloc[::-1],rank_cols,rank_headers))
        lines.append("")

        # alerts
        if alerts:
            lines.append(f">>> 异常预警 ({len(alerts)}条)")
            for a in alerts:
                lines.append(f"  [{a['type']}] {a['factor_name']}: {a['message']}")
            lines.append("")

        # group summaries
        group_cols = ["count","avg_net_excess","avg_ls_netret","avg_rankic","avg_net_excess_pct"]
        group_headers = ["count","avg_excess","avg_ls_net","avg_ic","excess_pct"]
        if not tag_summary.empty:
            lines.append(">>> 按Tag分类汇总")
            lines.append(self._fmt_table(tag_summary,group_cols,group_headers))
            lines.append("")

        if not cat_summary.empty:
            lines.append(">>> 按Category分类汇总")
            lines.append(self._fmt_table(cat_summary,group_cols,group_headers))
            lines.append("")

        # full detail
        full_cols = ["net_excess","net_excess_pct","ls_netret","ls_netret_pct","rankic","rankic_pct","long_turnover","coverage"]
        full_headers = ["net_excess","excess_pct","ls_netret","ls_pct","rankic","ic_pct","turnover","coverage"]
        lines.append(">>> 全因子明细")
        lines.append(self._fmt_table(details,full_cols,full_headers))

        return "\n".join(lines)

    def _fmt_table(self,df:pd.DataFrame,cols:list[str],headers:list[str])->str:
        name_w = max(len(str(idx)) for idx in df.index)+2
        col_w = 12

        header_line = f"{'name':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in headers)
        lines = [header_line,"-"*len(header_line)]

        for idx,row in df.iterrows():
            parts = [f"{str(idx):<{name_w}}"]
            for c in cols:
                v = row.get(c,np.nan)
                if pd.isna(v):
                    parts.append(f"{'N/A':>{col_w}}")
                elif "pct" in c or c=="count":
                    parts.append(f"{v:>{col_w}.0f}")
                elif "excess" in c or "ret" in c or "turnover" in c or "coverage" in c:
                    parts.append(f"{v:>{col_w}.2%}")
                elif "rankic" in c or "ic" in c:
                    parts.append(f"{v:>{col_w}.4f}")
                else:
                    parts.append(f"{v:>{col_w}.4f}")
            lines.append("".join(parts))

        return "\n".join(lines)

    # ----------------------------------------------------------
    # save
    # ----------------------------------------------------------
    def save(self,report_data:dict,path:str):
        txt = self.render_txt(report_data)
        with open(path,"w",encoding="utf-8") as f:
            f.write(txt)
        return path
