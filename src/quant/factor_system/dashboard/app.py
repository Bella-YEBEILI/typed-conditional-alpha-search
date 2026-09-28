import os
import sys
from functools import lru_cache
import numpy as np
import pandas as pd
from flask import Flask,Response,render_template,request

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
if project_root not in sys.path:
    sys.path.insert(0,project_root)

from quant.factor_system.factor_hub.main import FactorManager
from quant.factor_system.factor_hub.core.factor_performance_engine import FactorPerformanceEngine

app = Flask(__name__)
_manager = None


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


def get_manager()->FactorManager:
    global _manager
    if _manager is None:
        _manager = FactorManager()
    return _manager


def safe_float(v,default=None):
    if v is None or (isinstance(v,float) and np.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError,ValueError):
        return default


def fmt_pct(v,decimals=2):
    if v is None:
        return "-"
    return f"{v*100:.{decimals}f}%"


def fmt_num(v,decimals=2):
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def fmt_int(v):
    if v is None:
        return "-"
    return f"{v:.0f}"


app.jinja_env.globals.update(fmt_pct=fmt_pct,fmt_num=fmt_num,fmt_int=fmt_int)


def _get_period_ends(fm,section,period):
    """Collect available period_end dates from the first portrait that has data."""
    registry = fm.access.list_factors()
    for name in registry:
        try:
            portrait = fm.get_portrait(name)
        except Exception:
            continue
        sec = portrait.get(section,{})
        if not isinstance(sec,dict):
            continue
        df = sec.get(period)
        if df is not None and hasattr(df,"index") and len(df)>0:
            ends = sorted(df.index.tolist(),reverse=True)
            return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in ends]
    return []


def _extract_perf(portrait,section,period,period_end):
    """Extract perf dict from portrait given section/period/period_end."""
    sec = portrait.get(section,{})
    if not isinstance(sec,dict):
        return {},""
    if period=="all":
        data = sec.get("all",{})
        return (data if isinstance(data,dict) else {}),""
    df = sec.get(period)
    if df is None or not hasattr(df,"iloc") or len(df)==0:
        return {},""
    if period_end:
        target = pd.Timestamp(period_end)
        if target in df.index:
            return df.loc[target].to_dict(),pd.Timestamp(target).strftime("%Y-%m-%d")
        return {},period_end
    target = df.index[-1]
    return df.iloc[-1].to_dict(),pd.Timestamp(target).strftime("%Y-%m-%d")


@lru_cache(maxsize=4)
def _get_benchmark_daily_rets(benchmark):
    if benchmark!="zz1000":
        return pd.Series(dtype=float)
    fm = get_manager()
    index_data = fm.dm.get_data("index_data")
    daily_rets = index_data["zz1000s_returns"].dropna().copy()
    daily_rets.index = pd.to_datetime(daily_rets.index)
    return daily_rets.sort_index()


@lru_cache(maxsize=16)
def _get_benchmark_period_rets(benchmark,period):
    daily_rets = _get_benchmark_daily_rets(benchmark)
    if daily_rets.empty:
        return {}
    if period=="all":
        ret = FactorPerformanceEngine.agg_rets(daily_rets,243,annualize=True)[0]
        return {"all":ret}
    freq_map = {
        "weekly":"W",
        "monthly":"M",
        "yearly":"Y",
    }
    freq = freq_map.get(period)
    if freq is None:
        return {}
    data = {}
    for _,group in daily_rets.groupby(daily_rets.index.to_period(freq)):
        period_ret = FactorPerformanceEngine.agg_rets(group,243,annualize=False)[0]
        period_end = pd.Timestamp(group.index[-1]).strftime("%Y-%m-%d")
        data[period_end] = period_ret
    return data


def _get_benchmark_ret(benchmark,period,period_end):
    if benchmark=="":
        return None
    rets = _get_benchmark_period_rets(benchmark,period)
    key = "all" if period=="all" else period_end
    return rets.get(key)


def _slice_period_series(series,period,period_end):
    series = series.sort_index()
    if series.empty:
        return series
    if period=="all":
        return series
    freq_map = {
        "weekly":"W",
        "monthly":"M",
        "yearly":"Y",
    }
    freq = freq_map.get(period)
    if freq is None:
        return pd.Series(dtype=float)
    groups = list(series.groupby(series.index.to_period(freq)))
    if not groups:
        return pd.Series(dtype=float)
    if period_end:
        target = pd.Timestamp(period_end)
        for _,group in groups:
            if pd.Timestamp(group.index[-1])==target:
                return group
        return pd.Series(dtype=float)
    return groups[-1][1]


def _calc_maxdd(rets):
    if rets.empty:
        return None
    nav = (1+rets.fillna(0)).cumprod()
    dd = nav/nav.cummax()-1
    return safe_float(-dd.min())


def _get_excess_net_maxdd(fm,factor_name,period,period_end,benchmark):
    if benchmark=="":
        return None
    try:
        result = fm.access.get_result(factor_name)
    except Exception:
        return None
    long_rets = result.get("long_rets",pd.Series(dtype=float))
    long_turnovers = result.get("long_turnovers",pd.Series(dtype=float))
    long_netrets = long_rets-(long_turnovers/2*0.0012)
    long_netrets = _slice_period_series(long_netrets,period,period_end)
    if long_netrets.empty:
        return None
    benchmark_rets = _get_benchmark_daily_rets(benchmark)
    benchmark_rets,long_netrets = benchmark_rets.align(long_netrets,join="inner")
    if long_netrets.empty:
        return None
    return _calc_maxdd(long_netrets-benchmark_rets)


@app.route("/")
def index():
    fm = get_manager()
    registry = fm.access.list_factors()

    sort_by = request.args.get("sort_by","factor_name").strip()
    sort_dir = request.args.get("sort_dir","asc").strip()
    q = request.args.get("q","").strip().lower()

    tier = request.args.get("tier","selected").strip()
    if tier not in ("selected","pool","all"):
        tier = "selected"

    section = request.args.get("section","raw").strip()
    if section not in ("raw","hs300s","zz1000s","complete"):
        section = "raw"
    period = request.args.get("period","all").strip()
    if period not in ("all","yearly","monthly","weekly"):
        period = "all"
    period_end = request.args.get("period_end","").strip()

    benchmark_options = [("","None"),("zz1000","CSI 1000")]
    benchmark = request.args.get("benchmark","").strip()
    if benchmark not in {value for value,_ in benchmark_options}:
        benchmark = ""

    # collect available period_ends for dropdown
    period_ends = []
    if period!="all":
        period_ends = _get_period_ends(fm,section,period)

    factor_names = fm.access.interactor.filter_names(tier=tier)

    rows = []
    for name in factor_names:
        info = registry.get(name,{})
        try:
            portrait = fm.get_portrait(name)
        except Exception:
            portrait = {}

        perf,perf_period_end = _extract_perf(portrait,section,period,period_end)

        long_ret = safe_float(perf.get("long_ret"))
        long_netret = safe_float(perf.get("long_netret"))
        long_netmaxdd = safe_float(perf.get("long_netmaxdd"))
        benchmark_ret = _get_benchmark_ret(benchmark,period,perf_period_end)
        if benchmark_ret is not None:
            if long_ret is not None:
                long_ret = long_ret-benchmark_ret
            if long_netret is not None:
                long_netret = long_netret-benchmark_ret
        if benchmark:
            long_netmaxdd = _get_excess_net_maxdd(fm,name,period,perf_period_end,benchmark)

        row = {
            "factor_name":name,
            "author":info.get("author",""),
            "type":info.get("type",""),
            "level":info.get("level",""),
            "tag":info.get("tag",""),
            "category":info.get("category",""),
            "universe":info.get("universe",""),
            "ls_ir":safe_float(perf.get("ls_ir")),
            "ls_netir":safe_float(perf.get("ls_netir")),
            "long_ret":long_ret,
            "long_netret":long_netret,
            "long_netmaxdd":long_netmaxdd,
            "rankic":safe_float(perf.get("rankic")),
            "rankicir":safe_float(perf.get("rankicir")),
            "long_turnover":safe_float(perf.get("long_turnover")),
            "coverage":safe_float(perf.get("coverage")),
        }
        rows.append(row)

    if q:
        rows = [r for r in rows if q in r["factor_name"].lower()
                or q in (r["author"] or "").lower()
                or q in (r["tag"] or "").lower()
                or q in (r["category"] or "").lower()]

    allowed_sort = {
        "factor_name","author","tag","category",
        "ls_ir","ls_netir","long_ret","long_netret","long_netmaxdd","rankic","rankicir",
        "long_turnover","coverage",
    }
    if sort_by not in allowed_sort:
        sort_by = "factor_name"
    if sort_dir not in ("asc","desc"):
        sort_dir = "asc"

    non_none = [r for r in rows if r.get(sort_by) is not None]
    none_rows = [r for r in rows if r.get(sort_by) is None]

    if sort_by in ("factor_name","author","tag","category"):
        non_none.sort(key=lambda x:str(x.get(sort_by) or "").lower(),reverse=(sort_dir=="desc"))
    else:
        non_none.sort(key=lambda x:x.get(sort_by),reverse=(sort_dir=="desc"))

    rows = non_none+none_rows

    return render_template(
        "factor_list.html",
        factors=rows,
        q=request.args.get("q",""),
        sort_by=sort_by,
        sort_dir=sort_dir,
        tier=tier,
        section=section,
        period=period,
        period_end=period_end,
        period_ends=period_ends,
        benchmark=benchmark,
        benchmark_options=benchmark_options,
        long_ret_label="Long Excess Ret" if benchmark else "Long Ret",
        long_netret_label="Long Excess NetRet" if benchmark else "Long NetRet",
        long_netmaxdd_label="Excess NetMaxDD" if benchmark else "Long NetMaxDD",
    )


@app.route("/factor/<path:factor_name>")
def factor_detail(factor_name:str):
    fm = get_manager()

    info = fm.access.get_factor(factor_name)
    if info is None:
        return "factor not found",404

    try:
        portrait = fm.get_portrait(factor_name)
    except Exception:
        portrait = {}

    full_perf = portrait.get("raw",{}).get("all",{}) if isinstance(portrait.get("raw"),dict) else {}

    # performance with custom date range / annualize
    perf_start = request.args.get("perf_start","").strip()
    perf_end = request.args.get("perf_end","").strip()
    annualize = request.args.get("annualize","1").strip()!="0"
    perf_custom = bool(perf_start or perf_end or not annualize)

    if perf_custom:
        try:
            result_perf = fm.access.get_result(factor_name)
            params = {
                "start":perf_start or None,
                "end":perf_end or None,
                "cost":0.0012,
                "trading_days":243,
            }
            perf_engine = FactorPerformanceEngine()
            perf_data = perf_engine.calc_basic_performance(result_perf,params,annualize=annualize)
        except Exception:
            perf_data = full_perf
    else:
        perf_data = full_perf

    # result chart
    refresh_chart = request.args.get("refresh_chart","0")=="1"
    chart_type = request.args.get("chart_type","group_returns").strip()
    chart_start = request.args.get("chart_start","").strip()
    chart_end = request.args.get("chart_end","").strip()
    benchmark_options = [("zz1000","zz1000s")]
    benchmark = request.args.get("benchmark","zz1000").strip()
    if benchmark not in {value for value,_ in benchmark_options}:
        benchmark = "zz1000"
    chart_type_options = ["group_returns","group_returns_demeaned","ls","long","ex","rankic"]
    if chart_type not in chart_type_options:
        chart_type = "group_returns"

    chart_json = {"labels":[],"datasets":[]}
    chart_is_percent = True

    if refresh_chart:
        try:
            result = fm.access.get_result(factor_name)
        except Exception:
            result = {}

        if result:
            long_rets = result.get("long_rets",pd.Series(dtype=float))
            short_rets = result.get("short_rets",pd.Series(dtype=float))
            long_turnovers = result.get("long_turnovers",pd.Series(dtype=float))
            short_turnovers = result.get("short_turnovers",pd.Series(dtype=float))
            rankics_s = result.get("rankics",pd.Series(dtype=float))
            group_rets_df = result.get("group_rets",pd.DataFrame())

            cost = 0.0012
            long_netrets = long_rets-(long_turnovers/2*cost)
            short_netrets = short_rets-(short_turnovers/2*cost)
            ls_rets = (long_rets+short_rets)/2
            ls_netrets = (long_netrets+short_netrets)/2

            idx = long_rets.index
            if chart_start:
                idx = idx[idx>=chart_start]
            if chart_end:
                idx = idx[idx<=chart_end]

            labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in idx]

            def cumprod_list(s):
                s = s.reindex(idx).fillna(0)
                nav = (1+s).cumprod()-1
                return [safe_float(v) for v in nav.tolist()]

            def cumsum_list(s):
                s = s.reindex(idx).fillna(0)
                cs = s.cumsum()
                return [safe_float(v) for v in cs.tolist()]

            def pnl_list(s):
                s = s.reindex(idx).fillna(0)
                pnl = (1+s).cumprod()
                return [safe_float(v) for v in pnl.tolist()]

            benchmark_label = dict(benchmark_options).get(benchmark,benchmark)
            datasets = []
            if chart_type=="group_returns":
                for col in group_rets_df.columns:
                    datasets.append({"label":col,"data":cumprod_list(group_rets_df[col])})
                chart_is_percent = True
            elif chart_type=="group_returns_demeaned":
                gp = group_rets_df.reindex(idx).fillna(0)
                row_mean = gp.mean(axis=1)
                gp_dm = gp.sub(row_mean,axis=0)
                for col in gp_dm.columns:
                    datasets.append({"label":col,"data":cumsum_list(gp_dm[col])})
                chart_is_percent = True
            elif chart_type=="ls":
                datasets = [
                    {"label":"LS","data":cumprod_list(ls_rets)},
                    {"label":"LS Net","data":cumprod_list(ls_netrets)},
                ]
                chart_is_percent = True
            elif chart_type=="long":
                datasets = [
                    {"label":"Long","data":cumprod_list(long_rets)},
                    {"label":"Long Net","data":cumprod_list(long_netrets)},
                ]
                chart_is_percent = True
            elif chart_type=="ex":
                benchmark_rets = _get_benchmark_daily_rets(benchmark)
                ex_rets = long_netrets.reindex(idx).fillna(0)-benchmark_rets.reindex(idx).fillna(0)
                datasets = [
                    {"label":"Long Net PNL","data":pnl_list(long_netrets)},
                    {"label":f"{benchmark_label} PNL","data":pnl_list(benchmark_rets)},
                    {"label":"Excess PNL","data":pnl_list(ex_rets)},
                ]
                chart_is_percent = False
            elif chart_type=="rankic":
                datasets = [{"label":"RankIC CumSum","data":cumsum_list(rankics_s)}]
                chart_is_percent = False

            chart_json = {"labels":labels,"datasets":datasets}

    # correlation
    refresh_corr = request.args.get("refresh_corr","0")=="1"
    corr_info = {"max_corr":None,"avg_corr":None,"top_list":[]}

    if refresh_corr:
        try:
            result_corr = fm.access.get_result(factor_name)
            rankics_series = result_corr.get("rankics",pd.Series(dtype=float))
            rankics_data = fm.access.interactor.get_rankics_data(persist_result=False)
            corr_s = fm.access.comparator.check_prod_corr(lib_data=rankics_data,new_data=rankics_series)
            if factor_name in corr_s.index:
                corr_s = corr_s.drop(index=factor_name)
            corr_s = corr_s.dropna()
            if len(corr_s)>0:
                top_list = [{"factor_name":str(fn),"corr":safe_float(corr_s[fn])} for fn in corr_s.index]
                corr_info = {
                    "max_corr":safe_float(corr_s.abs().max()),
                    "avg_corr":safe_float(corr_s.mean()),
                    "top_list":top_list,
                }
        except Exception:
            pass

    return render_template(
        "factor_detail.html",
        info=info,
        factor_name=factor_name,
        perf_data=perf_data,
        safe_float=safe_float,
        perf_start=perf_start,
        perf_end=perf_end,
        annualize=annualize,
        refresh_chart=refresh_chart,
        chart_json=chart_json,
        chart_type=chart_type,
        chart_type_options=chart_type_options,
        benchmark=benchmark,
        benchmark_options=benchmark_options,
        chart_start=chart_start,
        chart_end=chart_end,
        chart_is_percent=chart_is_percent,
        refresh_corr=refresh_corr,
        corr_info=corr_info,
    )




def _get_available_weeks(fm):
    """Get all available week_end dates from portraits, newest first."""
    registry = fm.access.list_factors()
    for name in registry:
        try:
            portrait = fm.get_portrait(name)
        except Exception:
            continue
        raw = portrait.get("raw",{})
        if not isinstance(raw,dict):
            continue
        wp = raw.get("weekly")
        if wp is not None and hasattr(wp,"index") and len(wp)>0:
            ends = sorted(wp.index.tolist(),reverse=True)
            return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in ends]
    return []


@app.route("/weekly-report")
def weekly_report():
    fm = get_manager()
    week = request.args.get("week","").strip()

    available_weeks = _get_available_weeks(fm)

    end_date = week if week else None
    report = fm.reporter.generate(end_date)

    meta = report.get("meta",{})
    summary = report.get("summary",{})
    details = report.get("details",pd.DataFrame())
    alerts = report.get("alerts",[])
    tag_summary = report.get("tag_summary",pd.DataFrame())
    cat_summary = report.get("category_summary",pd.DataFrame())

    # convert details to list of dicts for template
    detail_rows = []
    if not details.empty:
        for name,row in details.iterrows():
            detail_rows.append({
                "factor_name":str(name),
                "net_excess":safe_float(row.get("net_excess")),
                "net_excess_pct":safe_float(row.get("net_excess_pct")),
                "ls_netret":safe_float(row.get("ls_netret")),
                "ls_netret_pct":safe_float(row.get("ls_netret_pct")),
                "rankic":safe_float(row.get("rankic")),
                "rankic_pct":safe_float(row.get("rankic_pct")),
                "long_turnover":safe_float(row.get("long_turnover")),
                "coverage":safe_float(row.get("coverage")),
                "tag":row.get("tag",""),
                "category":str(row.get("category","")),
            })

    # convert group summaries
    def df_to_rows(df):
        rows = []
        if df is not None and not df.empty:
            for name,row in df.iterrows():
                rows.append({
                    "name":str(name),
                    "count":int(row.get("count",0)),
                    "avg_net_excess":safe_float(row.get("avg_net_excess")),
                    "avg_ls_netret":safe_float(row.get("avg_ls_netret")),
                    "avg_rankic":safe_float(row.get("avg_rankic")),
                    "avg_net_excess_pct":safe_float(row.get("avg_net_excess_pct")),
                })
        return rows

    week_end_str = ""
    we = meta.get("week_end")
    if we is not None:
        week_end_str = pd.Timestamp(we).strftime("%Y-%m-%d")

    return render_template(
        "weekly_report.html",
        available_weeks=available_weeks,
        selected_week=week,
        week_end=week_end_str,
        factor_count=meta.get("factor_count",0),
        summary=summary,
        detail_rows=detail_rows,
        alerts=alerts,
        tag_rows=df_to_rows(tag_summary),
        cat_rows=df_to_rows(cat_summary),
    )

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5002,debug=False,use_reloader=False)
